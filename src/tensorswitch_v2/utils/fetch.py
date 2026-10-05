"""
Fetch remote microscopy files (HTTP/HTTPS, FTP, public S3) into a local folder.

A *spec* is either a plain URL or ``<zip url>::<member path>``, which pulls one
member out of a remote zip using range requests, so a 40 GB archive costs only
the bytes of the member (plus ~100 KB for the zip index).

Safety rules enforced here, so every caller (CLI, MCP) gets them:
- only http, https, ftp and s3 URLs;
- a byte cap (``max_bytes``) checked before and while downloading;
- destination names are confined to the destination folder (no zip-slip);
- nothing that is downloaded is executed or unpickled;
- files are written to ``<name>.part`` and renamed only when complete.
"""

import os
import re
import struct
import time
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urlsplit

import requests

ALLOWED_SCHEMES = ("http", "https", "ftp", "s3")
DEFAULT_MAX_BYTES = 2 * 1024 ** 3
USER_AGENT = "tensorswitch-fetch"
_ZIP_TAIL = 128 * 1024
_TIMEOUT = 120
DEFAULT_ATTEMPTS = 5     # a dropped connection is retried, continuing from the bytes already saved


class FetchError(Exception):
    """A download was refused or failed."""


@dataclass(frozen=True)
class ZipEntry:
    name: str
    size: int
    csize: int
    method: int
    offset: int
    crc: int


def parse_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Split ``<url>::<member>`` into (url, member); member is None for a plain URL."""
    spec = spec.strip()
    if "::" in spec:
        url, member = spec.split("::", 1)
        return url.strip(), member.strip() or None
    return spec, None


def normalize_url(url: str) -> str:
    """Validate the scheme and turn s3://bucket/key into its public HTTPS form."""
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise FetchError(f"unsupported URL scheme {scheme!r} (allowed: {', '.join(ALLOWED_SCHEMES)}): {url}")
    if not parts.netloc:
        raise FetchError(f"URL has no host: {url}")
    if scheme == "s3":
        return f"https://{parts.netloc}.s3.amazonaws.com/{parts.path.lstrip('/')}"
    return url


def safe_destination(dest_dir: str, relative_name: str) -> str:
    """Join a remote-supplied name onto dest_dir, refusing anything that escapes it."""
    name = relative_name.replace("\\", "/")
    if not name or name.endswith("/") or name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise FetchError(f"refusing unsafe file name: {relative_name!r}")
    root = os.path.realpath(dest_dir)
    target = os.path.realpath(os.path.join(root, *name.split("/")))
    if os.path.commonpath([root, target]) != root or target == root:
        raise FetchError(f"refusing file name that escapes the destination: {relative_name!r}")
    return target


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000


def _headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    return {"User-Agent": USER_AGENT, **(extra or {})}


def remote_size(url: str) -> Optional[int]:
    """Size in bytes of an HTTP(S) resource, or None if the server does not say."""
    resp = requests.head(url, headers=_headers(), timeout=_TIMEOUT, allow_redirects=True)
    size = int(resp.headers.get("Content-Length") or 0)
    if size:
        return size
    resp = requests.get(url, headers=_headers({"Range": "bytes=0-0"}), timeout=_TIMEOUT,
                        allow_redirects=True, stream=True)
    resp.close()
    total = resp.headers.get("Content-Range", "").rsplit("/", 1)[-1]
    return int(total) if total.isdigit() else None


def _range(url: str, start: int, end: int) -> bytes:
    resp = requests.get(url, headers=_headers({"Range": f"bytes={start}-{end}"}),
                        timeout=_TIMEOUT, allow_redirects=True)
    if resp.status_code != 206:
        raise FetchError(f"server does not support range requests (HTTP {resp.status_code}); "
                         f"cannot read part of {url} without downloading all of it")
    return resp.content


def list_zip(url: str) -> List[ZipEntry]:
    """Members of a remote zip, read from its central directory (HTTP range requests)."""
    url = normalize_url(url)
    if urlsplit(url).scheme == "ftp":
        raise FetchError("zip members cannot be read over FTP; use the https:// form of the URL")
    size = remote_size(url)
    if not size:
        raise FetchError(f"server did not report a size for {url}; cannot locate the zip index")
    tail = _range(url, max(0, size - _ZIP_TAIL), size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise FetchError(f"no zip end-of-central-directory record found; not a zip file? {url}")
    count, cd_size, cd_off = struct.unpack("<HII", tail[eocd + 10:eocd + 20])
    if cd_off == 0xFFFFFFFF or count == 0xFFFF:  # zip64
        loc = tail.rfind(b"PK\x06\x07")
        if loc < 0:
            raise FetchError(f"zip64 archive without a zip64 locator: {url}")
        z64_off = struct.unpack("<Q", tail[loc + 8:loc + 16])[0]
        rec = _range(url, z64_off, z64_off + 55)
        count, cd_size, cd_off = struct.unpack("<QQQ", rec[32:56])
    directory = _range(url, cd_off, cd_off + cd_size - 1)

    entries, i = [], 0
    while i + 46 <= len(directory) and directory[i:i + 4] == b"PK\x01\x02":
        flags, method = struct.unpack("<HH", directory[i + 8:i + 12])
        crc, csize, usize = struct.unpack("<III", directory[i + 16:i + 28])
        name_len, extra_len, comment_len = struct.unpack("<HHH", directory[i + 28:i + 34])
        offset = struct.unpack("<I", directory[i + 42:i + 46])[0]
        raw_name = directory[i + 46:i + 46 + name_len]
        name = raw_name.decode("utf-8") if flags & 0x800 else raw_name.decode("cp437")
        extra = directory[i + 46 + name_len:i + 46 + name_len + extra_len]
        if 0xFFFFFFFF in (usize, csize, offset):
            j = 0
            while j + 4 <= len(extra):
                tag, sz = struct.unpack("<HH", extra[j:j + 4])
                if tag == 1:
                    values = list(struct.unpack(f"<{sz // 8}Q", extra[j + 4:j + 4 + sz // 8 * 8]))
                    k = 0
                    if usize == 0xFFFFFFFF and k < len(values):
                        usize, k = values[k], k + 1
                    if csize == 0xFFFFFFFF and k < len(values):
                        csize, k = values[k], k + 1
                    if offset == 0xFFFFFFFF and k < len(values):
                        offset = values[k]
                    break
                j += 4 + sz
        if not name.endswith("/") and "__MACOSX/" not in name:
            entries.append(ZipEntry(name, usize, csize, method, offset, crc))
        i += 46 + name_len + extra_len + comment_len
    return entries


def _write_atomically(dest: str, chunks, max_bytes: int, what: str, crc: Optional[int] = None) -> int:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    written, running_crc = 0, 0
    try:
        with open(part, "wb") as handle:
            for chunk in chunks:
                written += len(chunk)
                if written > max_bytes:
                    raise FetchError(f"{what} is larger than the {_human(max_bytes)} limit; aborted")
                running_crc = zlib.crc32(chunk, running_crc)
                handle.write(chunk)
        if crc is not None and (running_crc & 0xFFFFFFFF) != crc:
            raise FetchError(f"checksum mismatch for {what}: download is corrupt")
        os.replace(part, dest)
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise
    return written


def _http_chunks(url: str, headers: Optional[Dict[str, str]] = None, expect_status: int = 200):
    resp = requests.get(url, headers=_headers(headers), timeout=_TIMEOUT, stream=True, allow_redirects=True)
    try:
        if resp.status_code != expect_status:
            raise FetchError(f"HTTP {resp.status_code} for {url}")
        yield from resp.iter_content(1 << 20)
    finally:
        resp.close()


def _http_stream(url: str, offset: int):
    """Open url from byte `offset`. Returns (start, chunks, expected_total); start is 0 if the server ignores Range."""
    headers = {"Range": f"bytes={offset}-"} if offset else None
    resp = requests.get(url, headers=_headers(headers), timeout=_TIMEOUT, stream=True, allow_redirects=True)
    try:
        if offset and resp.status_code == 416:       # asked past the end: the saved part is complete or stale
            if remote_size(url) == offset:
                resp.close()
                return offset, iter(()), offset
            offset = 0
            resp.close()
            return _http_stream(url, 0)
        if resp.status_code == 206 and offset:
            start = offset
        elif resp.status_code == 200:
            start = 0
        else:
            raise FetchError(f"HTTP {resp.status_code} for {url}")
        length = resp.headers.get("Content-Length")
        expected = start + int(length) if length and not resp.headers.get("Content-Encoding") else None
    except BaseException:
        resp.close()
        raise

    def chunks():
        try:
            yield from resp.iter_content(1 << 20)
        finally:
            resp.close()
    return start, chunks(), expected


def _ftp_stream(url: str, offset: int):
    import ftplib

    parts = urlsplit(url)
    ftp = ftplib.FTP(parts.hostname, timeout=_TIMEOUT)
    try:
        ftp.login(parts.username or "anonymous", parts.password or "anonymous@")
        ftp.voidcmd("TYPE I")
        try:
            expected = ftp.size(unquote(parts.path))
        except ftplib.all_errors:
            expected = None
        start = offset
        try:
            conn = ftp.transfercmd(f"RETR {unquote(parts.path)}", rest=offset or None)
        except ftplib.error_perm:
            if not offset:
                raise
            start = 0                                 # server cannot restart mid-file
            conn = ftp.transfercmd(f"RETR {unquote(parts.path)}")
    except BaseException:
        try:
            ftp.close()
        except Exception:
            pass
        raise

    def chunks():
        try:
            with conn:
                while True:
                    data = conn.recv(1 << 20)
                    if not data:
                        break
                    yield data
            ftp.voidresp()
        finally:
            try:
                ftp.quit()
            except Exception:
                pass
    return start, chunks(), expected


def _transient_errors():
    import ftplib

    return (requests.RequestException, OSError, EOFError, ftplib.error_temp, ftplib.error_reply)


def _download_resumable(open_stream, dest: str, max_bytes: int, what: str,
                        attempts: int = DEFAULT_ATTEMPTS, progress=None) -> int:
    """Download to ``dest + '.part'``, continuing from the saved bytes after a dropped connection."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    transient = _transient_errors()
    for attempt in range(1, attempts + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        try:
            start, chunks, expected = open_stream(have)
            written = start if start == have else 0
            with open(part, "ab" if start == have and have else "wb") as handle:
                for chunk in chunks:
                    written += len(chunk)
                    if written > max_bytes:
                        raise FetchError(f"{what} is larger than the {_human(max_bytes)} limit; aborted")
                    handle.write(chunk)
                    if progress:
                        progress(written, expected)
            if expected is not None and written != expected:
                raise ConnectionError(f"connection closed after {written} of {expected} bytes")
            os.replace(part, dest)
            return written
        except FetchError:
            if os.path.exists(part):
                os.remove(part)
            raise
        except transient as err:
            if attempt == attempts:
                saved = os.path.getsize(part) if os.path.exists(part) else 0
                raise FetchError(f"download of {what} interrupted ({err}); {_human(saved)} saved in {part}. "
                                 f"Run it again to continue from there.") from err
            time.sleep(min(2 ** attempt, 30))
    raise AssertionError("unreachable")


def fetch_file(url: str, dest_dir: str, max_bytes: int = DEFAULT_MAX_BYTES,
               filename: Optional[str] = None, attempts: int = DEFAULT_ATTEMPTS, progress=None) -> Dict:
    """Download one whole file into dest_dir. Returns {'path', 'bytes', 'source'}.

    A download that is cut off keeps its ``.part`` file; running the same call again
    (or the automatic retries) continues from there when the server allows it.
    """
    url = normalize_url(url)
    scheme = urlsplit(url).scheme
    name = filename or unquote(os.path.basename(urlsplit(url).path))
    dest = safe_destination(dest_dir, name)
    if scheme != "ftp":
        size = remote_size(url)
        if size and size > max_bytes:
            raise FetchError(f"{name} is {_human(size)}, over the {_human(max_bytes)} limit")
    stream = _ftp_stream if scheme == "ftp" else _http_stream
    written = _download_resumable(lambda offset: stream(url, offset), dest, max_bytes, name, attempts, progress)
    return {"path": dest, "bytes": written, "source": url}


def expected_path(spec: str, dest_dir: str) -> str:
    """Where fetch() will put a spec inside dest_dir."""
    url, member = parse_spec(spec)
    name = member if member else unquote(os.path.basename(urlsplit(normalize_url(url)).path))
    return safe_destination(dest_dir, name)


def fetch_zip_member(url: str, member: str, dest_dir: str, max_bytes: int = DEFAULT_MAX_BYTES) -> Dict:
    """Download one member of a remote zip into dest_dir, keeping its path inside the zip."""
    url = normalize_url(url)
    entries = {e.name: e for e in list_zip(url)}
    entry = entries.get(member)
    if entry is None:
        close = [n for n in entries if n.endswith("/" + member) or member.endswith("/" + n)][:3]
        hint = f" Did you mean: {close}?" if close else ""
        raise FetchError(f"{member!r} is not in the zip ({len(entries)} members).{hint}")
    if entry.method not in (0, 8):
        raise FetchError(f"unsupported zip compression method {entry.method} for {member}")
    if entry.size > max_bytes:
        raise FetchError(f"{member} is {_human(entry.size)} uncompressed, over the {_human(max_bytes)} limit")
    dest = safe_destination(dest_dir, entry.name)

    local = _range(url, entry.offset, entry.offset + 29)
    if local[:4] != b"PK\x03\x04":
        raise FetchError(f"bad zip local header for {member}")
    name_len, extra_len = struct.unpack("<HH", local[26:30])
    start = entry.offset + 30 + name_len + extra_len
    raw = _http_chunks(url, {"Range": f"bytes={start}-{start + entry.csize - 1}"}, expect_status=206) \
        if entry.csize else iter(())

    def decoded():
        if entry.method == 0:
            yield from raw
            return
        inflater = zlib.decompressobj(-15)
        for block in raw:
            out = inflater.decompress(block)
            if out:
                yield out
        tail = inflater.flush()
        if tail:
            yield tail

    written = _write_atomically(dest, decoded(), max_bytes, member, crc=entry.crc)
    if written != entry.size:
        os.remove(dest)
        raise FetchError(f"{member}: expected {entry.size} bytes, got {written}")
    return {"path": dest, "bytes": written, "source": f"{url}::{member}"}


def fetch(spec: str, dest_dir: str, max_bytes: int = DEFAULT_MAX_BYTES, progress=None) -> Dict:
    """Fetch a spec (``url`` or ``zip url::member``) into dest_dir."""
    url, member = parse_spec(spec)
    if member:
        return fetch_zip_member(url, member, dest_dir, max_bytes)
    return fetch_file(url, dest_dir, max_bytes, progress=progress)


DEFAULT_ALLOWED_HOSTS = (
    "zenodo.org", "ftp.ebi.ac.uk", "www.ebi.ac.uk", "data.broadinstitute.org",
    "data.celltrackingchallenge.net", "ndownloader.figshare.com", "figshare.com",
    "huggingface.co", "github.com", "raw.githubusercontent.com", "s3.amazonaws.com",
    "datasets.gryf.fi.muni.cz", "rgw.cscs.ch", "files.cryoetdataportal.cziscience.com",
    "dataverse.harvard.edu", "data.mendeley.com", "datadryad.org", "osf.io",
    "bossdb-open-data.s3.amazonaws.com", "janelia-cosem-datasets.s3.amazonaws.com",
)


def allowed_hosts() -> Tuple[str, ...]:
    """Hosts agents may fetch from: defaults plus TENSORSWITCH_FETCH_HOSTS (comma separated)."""
    extra = tuple(h.strip().lower() for h in os.environ.get("TENSORSWITCH_FETCH_HOSTS", "").split(",") if h.strip())
    return DEFAULT_ALLOWED_HOSTS + extra


def check_host(url: str) -> None:
    """Raise FetchError unless the URL's host (or a parent domain) is allowed.

    Only the first host is checked; servers may redirect elsewhere (Zenodo does).
    """
    host = (urlsplit(normalize_url(url)).hostname or "").lower()
    if not any(host == h or host.endswith("." + h) for h in allowed_hosts()):
        raise FetchError(
            f"host {host!r} is not on the fetch allowlist. Download it yourself, or set "
            f"TENSORSWITCH_FETCH_HOSTS={host} to allow it."
        )


# ----------------------------------------------------------------------------- background

_CHILDREN: Dict[int, "object"] = {}     # processes started here, so finished ones can be reaped


def _pid_running(pid: int) -> bool:
    child = _CHILDREN.get(pid)
    if child is not None:
        return child.poll() is None
    try:
        with open(f"/proc/{pid}/stat") as handle:
            if handle.read().rsplit(")", 1)[-1].split()[0] == "Z":
                return False
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return b"tensorswitch_v2.utils.fetch" in handle.read()
    except FileNotFoundError:
        return False
    except OSError:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _last_line(path: str) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 4096))
            lines = [l.strip() for l in handle.read().decode("utf-8", "replace").splitlines() if l.strip()]
        return lines[-1] if lines else ""
    except OSError:
        return ""


def background_fetch(spec: str, dest_dir: str, max_bytes: int = DEFAULT_MAX_BYTES) -> Dict:
    """Start the download in a detached process, or report on the one already started.

    Safe to call again with the same arguments: it returns the finished file, the progress
    of a running download, the error of a failed one (the next call tries again), or starts
    a new process that continues from any saved ``.part`` file.
    Returns a dict with ``state`` in done / downloading / started / failed.
    """
    import json
    import subprocess
    import sys

    dest = expected_path(spec, dest_dir)
    if os.path.isfile(dest):
        return {"state": "done", "path": dest, "bytes": os.path.getsize(dest)}
    folder, name = os.path.dirname(dest), os.path.basename(dest)
    os.makedirs(folder, exist_ok=True)
    log, marker = os.path.join(folder, f".{name}.fetch.log"), os.path.join(folder, f".{name}.fetch.json")
    saved = os.path.getsize(dest + ".part") if os.path.exists(dest + ".part") else 0

    if os.path.exists(marker):
        try:
            with open(marker) as handle:
                pid = json.load(handle)["pid"]
        except (OSError, ValueError, KeyError):
            pid = None
        if pid and _pid_running(pid):
            return {"state": "downloading", "path": dest, "pid": pid, "log": log,
                    "bytes_saved": saved, "last_message": _last_line(log)}
        os.remove(marker)
        last = _last_line(log)
        if last.startswith("error:"):
            return {"state": "failed", "path": dest, "log": log, "bytes_saved": saved, "message": last[6:].strip()}

    command = [sys.executable, "-m", "tensorswitch_v2.utils.fetch", spec, dest_dir, "--max-gb", f"{max_bytes / 1024 ** 3:g}"]
    with open(log, "w") as handle:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    _CHILDREN[child.pid] = child
    with open(marker, "w") as handle:
        json.dump({"pid": child.pid, "spec": spec}, handle)
    return {"state": "started", "path": dest, "pid": child.pid, "log": log, "bytes_saved": saved}


def main(argv=None) -> int:
    """``python -m tensorswitch_v2.utils.fetch <spec> <dest_dir> [--max-gb N]``"""
    import argparse

    parser = argparse.ArgumentParser(description="Download a URL or a zip member (<zip url>::<member>).")
    parser.add_argument("spec")
    parser.add_argument("dest_dir")
    parser.add_argument("--max-gb", type=float, default=DEFAULT_MAX_BYTES / 1024 ** 3,
                        help="refuse anything larger (default: %(default).0f GB)")
    args = parser.parse_args(argv)
    last = [0.0]

    def progress(written, total):
        now = time.monotonic()
        if now - last[0] >= 10:
            last[0] = now
            print(f"downloaded {_human(written)}" + (f" of {_human(total)}" if total else ""), flush=True)

    try:
        result = fetch(args.spec, args.dest_dir, int(args.max_gb * 1024 ** 3), progress=progress)
    except FetchError as err:
        print(f"error: {err}")
        return 1
    print(f"saved {result['path']} ({_human(result['bytes'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
