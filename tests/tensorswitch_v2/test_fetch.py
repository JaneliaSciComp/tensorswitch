"""
Tests for utils/fetch.py: whole files and single zip members over HTTP, using a
local server that supports range requests (no internet needed).
"""

import os
import re
import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tensorswitch_v2.utils.fetch import (
    FetchError, fetch, fetch_file, fetch_zip_member, list_zip, normalize_url,
    parse_spec, safe_destination,
)


class RangeHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler plus single-range support (no extension)."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        header = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not header or not os.path.isfile(path):
            return super().do_GET()
        start, end = re.match(r"bytes=(\d+)-(\d*)", header).groups()
        size = os.path.getsize(path)
        start = int(start)
        end = min(int(end), size - 1) if end else size - 1
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as handle:
            handle.seek(start)
            self.wfile.write(handle.read(end - start + 1))


@pytest.fixture
def server(temp_dir):
    root = os.path.join(temp_dir, "www")
    os.makedirs(root)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=root))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture
def dest(temp_dir):
    path = os.path.join(temp_dir, "dest")
    os.makedirs(path)
    return path


def _make_zip(root, name="data.zip"):
    path = os.path.join(root, name)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("set/images/a.tif", b"A" * 5000, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("set/masks/a.tif", bytes(range(256)) * 40, compress_type=zipfile.ZIP_STORED)
        z.writestr("set/readme.txt", b"hello", compress_type=zipfile.ZIP_DEFLATED)
    return path


class TestHelpers:
    def test_parse_spec(self):
        assert parse_spec("http://h/a.zip::x/y.tif") == ("http://h/a.zip", "x/y.tif")
        assert parse_spec("http://h/a.tif") == ("http://h/a.tif", None)

    def test_s3_becomes_public_https(self):
        assert normalize_url("s3://bucket/a/b.tif") == "https://bucket.s3.amazonaws.com/a/b.tif"

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "gs://b/x", "javascript:alert(1)", "http://"])
    def test_rejected_urls(self, url):
        with pytest.raises(FetchError):
            normalize_url(url)

    @pytest.mark.parametrize("name", ["../evil", "a/../../evil", "/abs/path", "C:\\x", "dir/", "", "..\\evil"])
    def test_zip_slip_names_are_refused(self, dest, name):
        with pytest.raises(FetchError):
            safe_destination(dest, name)

    def test_nested_safe_name_is_allowed(self, dest):
        assert safe_destination(dest, "a/b/c.tif").startswith(os.path.realpath(dest))

    def test_symlink_escape_is_refused(self, dest, temp_dir):
        outside = os.path.join(temp_dir, "outside")
        os.makedirs(outside)
        os.symlink(outside, os.path.join(dest, "link"))
        with pytest.raises(FetchError):
            safe_destination(dest, "link/evil.tif")


class TestFetchFile:
    def test_downloads_and_names_file_from_url(self, server, dest):
        root, base = server
        with open(os.path.join(root, "vol.tif"), "wb") as f:
            f.write(b"x" * 3000)
        result = fetch_file(f"{base}/vol.tif", dest)
        assert os.path.basename(result["path"]) == "vol.tif"
        assert os.path.getsize(result["path"]) == 3000 == result["bytes"]
        assert not os.path.exists(result["path"] + ".part")

    def test_size_cap_is_enforced_before_download(self, server, dest):
        root, base = server
        with open(os.path.join(root, "big.bin"), "wb") as f:
            f.write(b"x" * 5000)
        with pytest.raises(FetchError, match="limit"):
            fetch_file(f"{base}/big.bin", dest, max_bytes=1000)
        assert os.listdir(dest) == []

    def test_http_error_leaves_no_partial_file(self, server, dest):
        _, base = server
        with pytest.raises(FetchError, match="404"):
            fetch_file(f"{base}/missing.tif", dest)
        assert os.listdir(dest) == []

    def test_url_encoded_name_is_decoded(self, server, dest):
        root, base = server
        with open(os.path.join(root, "my vol.tif"), "wb") as f:
            f.write(b"y")
        assert os.path.basename(fetch_file(f"{base}/my%20vol.tif", dest)["path"]) == "my vol.tif"


class TestZipMembers:
    def test_lists_members(self, server):
        root, base = server
        _make_zip(root)
        names = sorted(e.name for e in list_zip(f"{base}/data.zip"))
        assert names == ["set/images/a.tif", "set/masks/a.tif", "set/readme.txt"]

    @pytest.mark.parametrize("member,expected", [
        ("set/images/a.tif", b"A" * 5000),
        ("set/masks/a.tif", bytes(range(256)) * 40),
        ("set/readme.txt", b"hello"),
    ])
    def test_fetches_deflated_and_stored_members(self, server, dest, member, expected):
        root, base = server
        _make_zip(root)
        result = fetch_zip_member(f"{base}/data.zip", member, dest)
        assert open(result["path"], "rb").read() == expected
        assert result["path"] == os.path.join(os.path.realpath(dest), *member.split("/"))

    def test_only_the_member_is_transferred(self, server, dest):
        root, base = server
        with zipfile.ZipFile(os.path.join(root, "big.zip"), "w") as z:
            z.writestr("small.txt", b"s" * 10, compress_type=zipfile.ZIP_STORED)
            z.writestr("huge.bin", os.urandom(2_000_000), compress_type=zipfile.ZIP_STORED)
        result = fetch(f"{base}/big.zip::small.txt", dest)
        assert result["bytes"] == 10

    def test_member_over_cap_is_refused(self, server, dest):
        root, base = server
        _make_zip(root)
        with pytest.raises(FetchError, match="limit"):
            fetch_zip_member(f"{base}/data.zip", "set/images/a.tif", dest, max_bytes=100)
        assert os.listdir(dest) == []

    def test_unknown_member_suggests_close_match(self, server, dest):
        root, base = server
        _make_zip(root)
        with pytest.raises(FetchError, match="images/a.tif"):
            fetch_zip_member(f"{base}/data.zip", "images/a.tif", dest)

    def test_malicious_member_name_cannot_escape(self, server, dest, temp_dir):
        root, base = server
        with zipfile.ZipFile(os.path.join(root, "evil.zip"), "w") as z:
            z.writestr("../../escaped.txt", b"pwned")
        with pytest.raises(FetchError, match="escapes|unsafe"):
            fetch_zip_member(f"{base}/evil.zip", "../../escaped.txt", dest)
        assert not os.path.exists(os.path.join(temp_dir, "escaped.txt"))
        assert not os.path.exists(os.path.join(os.path.dirname(dest), "escaped.txt"))

    def test_corrupt_data_is_detected_and_not_kept(self, server, dest):
        root, base = server
        path = os.path.join(root, "bad.zip")
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("f.bin", b"0123456789" * 50, compress_type=zipfile.ZIP_STORED)
        data = bytearray(open(path, "rb").read())
        data[60] ^= 0xFF  # flip a byte inside the stored member
        open(path, "wb").write(bytes(data))
        with pytest.raises(FetchError, match="checksum"):
            fetch_zip_member(f"{base}/bad.zip", "f.bin", dest)
        assert os.listdir(dest) == [] or not any(
            n.endswith("f.bin") for _, _, fs in os.walk(dest) for n in fs)

    def test_not_a_zip(self, server, dest):
        root, base = server
        open(os.path.join(root, "plain.bin"), "wb").write(b"not a zip" * 100)
        with pytest.raises(FetchError, match="zip"):
            list_zip(f"{base}/plain.bin")

    def test_ftp_zip_members_are_refused(self):
        with pytest.raises(FetchError, match="FTP"):
            list_zip("ftp://host/a.zip")


class TestAllowlistAndCli:
    def test_default_host_is_allowed(self):
        from tensorswitch_v2.utils.fetch import check_host

        check_host("https://zenodo.org/records/1/files/a.zip")
        check_host("https://janelia-cosem-datasets.s3.amazonaws.com/x")
        check_host("s3://allencell/aics/x")  # becomes allencell.s3.amazonaws.com

    @pytest.mark.parametrize("url", [
        "http://169.254.169.254/latest/meta-data", "http://localhost:8000/x",
        "https://evil.example.com/a.tif", "https://zenodo.org.evil.com/a.tif",
    ])
    def test_other_hosts_are_refused(self, url):
        from tensorswitch_v2.utils.fetch import check_host

        with pytest.raises(FetchError, match="allowlist"):
            check_host(url)

    def test_env_var_extends_allowlist(self, monkeypatch):
        from tensorswitch_v2.utils.fetch import check_host

        monkeypatch.setenv("TENSORSWITCH_FETCH_HOSTS", "data.mylab.org")
        check_host("https://data.mylab.org/a.tif")

    def test_cli_downloads_and_reports_errors(self, server, dest, capsys):
        from tensorswitch_v2.utils.fetch import main

        root, base = server
        open(os.path.join(root, "a.bin"), "wb").write(b"z" * 10)
        assert main([f"{base}/a.bin", dest]) == 0
        assert "saved" in capsys.readouterr().out
        assert main([f"{base}/missing.bin", dest]) == 1


class TestMcpFetchTool:
    @pytest.fixture
    def tool(self, monkeypatch):
        pytest.importorskip("mcp")
        from tensorswitch_v2 import mcp_server

        monkeypatch.setenv("TENSORSWITCH_FETCH_HOSTS", "127.0.0.1")
        return lambda *a, **k: __import__("json").loads(mcp_server.fetch_dataset(*a, **k))

    def test_downloads_a_zip_member(self, tool, server, dest):
        root, base = server
        _make_zip(root)
        result = tool(f"{base}/data.zip::set/readme.txt", dest)
        assert result["status"] == "success" and open(result["path"], "rb").read() == b"hello"

    def test_refuses_unlisted_host(self, tool, dest):
        result = tool("https://evil.example.com/a.tif", dest)
        assert result["error"] == "fetch_refused" and "allowlist" in result["message"]

    def test_large_request_points_to_background_mode(self, tool, dest):
        result = tool("https://zenodo.org/records/1/files/big.zip", dest, max_gb=50)
        assert result["error"] == "too_large_for_mcp" and "background=True" in result["message"]
        assert "tensorswitch_v2.utils.fetch" in result["command"] and "bsub" not in result["command"]

    def test_size_cap_error_is_reported(self, tool, server, dest):
        root, base = server
        open(os.path.join(root, "b.bin"), "wb").write(b"x" * 5000)
        result = tool(f"{base}/b.bin", dest, max_gb=0.000001)
        assert result["error"] == "fetch_refused" and "limit" in result["message"]
