"""Downloads that are cut off continue from the bytes already saved (local server, no internet)."""
import os
import threading
import time
from functools import partial
from http.server import ThreadingHTTPServer

import pytest

from tensorswitch_v2.utils import fetch as fetch_module
from tensorswitch_v2.utils.fetch import FetchError, expected_path, fetch_file

from test_fetch import RangeHandler

PAYLOAD = bytes(range(256)) * 4000      # ~1 MB


class FlakyHandler(RangeHandler):
    """Cuts the first `drops` full downloads after half the bytes, then behaves."""
    drops = 1
    ignore_range = False
    seen = []

    def do_GET(self):
        type(self).seen.append(self.headers.get("Range"))
        path = self.translate_path(self.path)
        if self.ignore_range or (not self.headers.get("Range") and type(self).drops > 0 and os.path.isfile(path)):
            size = os.path.getsize(path)
            if not self.ignore_range:
                type(self).drops -= 1
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with open(path, "rb") as handle:
                data = handle.read(size if self.ignore_range else size // 2)
            self.wfile.write(data)
            self.wfile.flush()
            if not self.ignore_range:
                self.close_connection = True
                self.connection.close()
            return
        return super().do_GET()

    def do_HEAD(self):
        return super().do_HEAD()


@pytest.fixture
def flaky(temp_dir, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    root = os.path.join(temp_dir, "www")
    os.makedirs(root)
    with open(os.path.join(root, "big.bin"), "wb") as handle:
        handle.write(PAYLOAD)

    def make(drops=1, ignore_range=False):
        handler = type("H", (FlakyHandler,), {"drops": drops, "ignore_range": ignore_range, "seen": []})
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(handler, directory=root))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        make.servers.append(httpd)
        return f"http://127.0.0.1:{httpd.server_address[1]}/big.bin", handler

    make.servers = []
    yield make
    for httpd in make.servers:
        httpd.shutdown()


@pytest.fixture
def dest(temp_dir):
    path = os.path.join(temp_dir, "dest")
    os.makedirs(path)
    return path


def test_dropped_connection_resumes_from_the_saved_bytes(flaky, dest):
    url, handler = flaky(drops=1)
    result = fetch_file(url, dest)
    assert open(result["path"], "rb").read() == PAYLOAD
    assert handler.seen[0] is None and handler.seen[1].startswith("bytes=") and handler.seen[1] != "bytes=0-"
    assert not os.path.exists(result["path"] + ".part")


def test_server_that_ignores_range_restarts_cleanly(flaky, dest):
    url, handler = flaky(ignore_range=True)
    with open(os.path.join(dest, "big.bin.part"), "wb") as handle:
        handle.write(PAYLOAD[:1000])          # stale partial from an earlier run
    assert open(fetch_file(url, dest)["path"], "rb").read() == PAYLOAD


def test_existing_partial_file_is_continued(flaky, dest):
    url, handler = flaky(drops=0)
    with open(os.path.join(dest, "big.bin.part"), "wb") as handle:
        handle.write(PAYLOAD[:300000])
    assert open(fetch_file(url, dest)["path"], "rb").read() == PAYLOAD
    assert handler.seen == ["bytes=300000-"]


def test_partial_file_that_is_already_complete_is_finalized(flaky, dest):
    url, _ = flaky(drops=0)
    with open(os.path.join(dest, "big.bin.part"), "wb") as handle:
        handle.write(PAYLOAD)
    assert open(fetch_file(url, dest)["path"], "rb").read() == PAYLOAD


def test_gives_up_after_the_attempts_and_keeps_the_part(flaky, dest):
    url, _ = flaky(drops=10)
    with pytest.raises(FetchError, match="interrupted.*saved in .*big.bin.part"):
        fetch_file(url, dest, attempts=1)
    assert os.path.getsize(os.path.join(dest, "big.bin.part")) == len(PAYLOAD) // 2


def test_size_cap_removes_the_partial_file(flaky, dest):
    url, _ = flaky(drops=0, ignore_range=True)
    with pytest.raises(FetchError, match="limit|over the"):
        fetch_file(url, dest, max_bytes=1000)
    assert not os.path.exists(os.path.join(dest, "big.bin.part"))


def test_expected_path_matches_where_fetch_saves(dest):
    assert expected_path("https://zenodo.org/records/1/files/a%20b.tif", dest) == os.path.join(dest, "a b.tif")
    assert expected_path("https://zenodo.org/x/d.zip::set/img/a.tif", dest) == os.path.join(dest, "set", "img", "a.tif")


class TestBackgroundFetch:
    def _wait(self, call, wanted, seconds=30):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            state = call()
            if state["state"] in wanted:
                return state
            threading.Event().wait(0.2)
        raise AssertionError(f"still {state['state']}: {state}")

    @pytest.fixture(autouse=True)
    def _allow_local_host(self, monkeypatch):
        monkeypatch.setenv("TENSORSWITCH_FETCH_HOSTS", "127.0.0.1")

    def test_download_runs_detached_and_the_second_call_reports_it(self, flaky, dest):
        url, _ = flaky(drops=0)
        first = fetch_module.background_fetch(url, dest)
        assert first["state"] in ("started", "downloading", "done") and first["path"].endswith("big.bin")
        done = self._wait(lambda: fetch_module.background_fetch(url, dest), ("done",))
        assert open(done["path"], "rb").read() == PAYLOAD
        assert not [n for n in os.listdir(dest) if n.endswith(".part")]

    def test_a_failed_download_is_reported_once_then_retried(self, flaky, dest):
        url, _ = flaky(drops=0)
        fetch_module.background_fetch(url, dest, max_bytes=1000)     # over the cap
        failed = self._wait(lambda: fetch_module.background_fetch(url, dest, max_bytes=1000), ("failed",))
        assert "limit" in failed["message"] or "over the" in failed["message"]
        retry = fetch_module.background_fetch(url, dest)             # state was cleared: starts again
        assert retry["state"] in ("started", "done")
        self._wait(lambda: fetch_module.background_fetch(url, dest), ("done",))

    def test_mcp_tool_reports_status_values(self, flaky, dest):
        pytest.importorskip("mcp")
        import json
        from tensorswitch_v2 import mcp_server

        url, _ = flaky(drops=0)
        first = json.loads(mcp_server.fetch_dataset(url, dest, max_gb=5, background=True))
        assert first["status"] in ("started", "downloading", "success")
        final = self._wait(lambda: {"state": json.loads(mcp_server.fetch_dataset(url, dest, max_gb=5, background=True))["status"]},
                           ("success",))
        assert final["state"] == "success"
