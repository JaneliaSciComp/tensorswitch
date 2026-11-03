"""
Report on LSF jobs in a way that tells a real success from a silent no-op.

"DONE" only means the job's command exited 0. A job whose command did nothing (a shell
quoting problem, a wrapper that never started Python) is DONE too. So each job is
reported with its exit code, CPU time and peak memory, the tail of its stdout and
stderr logs, and flags for the cases that look wrong. Jobs that belong to it (a
conversion's pyramid coordinator, and the level jobs it starts) are included, so a
chain is reported as one.
"""

import os
import re
import subprocess
from typing import Any, Callable, Dict, List, Optional

FIELDS = "jobid job_name stat exit_code run_time cpu_used max_mem output_file error_file dependency"
ACTIVE = ("PEND", "RUN", "PSUSP", "USUSP", "SSUSP", "WAIT")
# A TensorSwitch job that really ran has started Python and imported TensorStore, which
# costs about a second of CPU even for a tiny job (the smallest of 216 finished jobs used
# 0.9 s). The silent no-op seen in practice (a shell wrapper that never started Python)
# used 0.1 s. Memory is not a usable signal: small level jobs legitimately use ~35 MB.
NOOP_CPU_SECONDS = 0.5


def _seconds(text: str) -> Optional[float]:
    m = re.match(r"\s*([\d.]+)\s*second", text or "")
    return float(m.group(1)) if m else None


def _megabytes(text: str) -> Optional[float]:
    m = re.match(r"\s*([\d.]+)\s*([KMGT])bytes", text or "")
    if not m:
        return None
    return float(m.group(1)) * {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 ** 2}[m.group(2)]


def _run_bjobs(args: List[str], runner: Callable) -> List[Dict[str, str]]:
    result = runner(["bjobs", "-a", "-noheader", "-o", FIELDS + " delimiter='|'", *args],
                    capture_output=True, text=True, timeout=60)
    rows = []
    for line in (result.stdout or "").splitlines():
        parts = line.split("|")
        if len(parts) == len(FIELDS.split()):
            rows.append(dict(zip(FIELDS.split(), parts)))
    return rows


def tail_of_log(path: Optional[str], lines: int = 15, drop_lsf_summary: bool = True) -> List[str]:
    """Last lines of a job log; the LSF resource summary appended to stdout logs is dropped."""
    if not path or path == "-" or not os.path.isfile(path):
        return []
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 64 * 1024))
            text = handle.read().decode("utf-8", "replace")
    except OSError:
        return []
    if drop_lsf_summary:
        text = re.split(r"\n?-{20,}\s*\nSender: LSF System", text, maxsplit=1)[0]
    kept = [l.rstrip() for l in text.splitlines() if l.strip()]
    return kept[-lines:]


def describe(row: Dict[str, str], log_lines: int = 15) -> Dict[str, Any]:
    """One job as a dict with flags for what looks wrong."""
    status = row["stat"]
    run, cpu, mem = _seconds(row["run_time"]), _seconds(row["cpu_used"]), _megabytes(row["max_mem"])
    out_tail = tail_of_log(row["output_file"], log_lines)
    err_tail = tail_of_log(row["error_file"], log_lines, drop_lsf_summary=False)
    name = row["job_name"]
    flags = []
    if status == "EXIT":
        flags.append(f"job failed (exit code {row['exit_code']})")
    if status == "DONE":
        if name.startswith("tsv2_") and cpu is not None and cpu < NOOP_CPU_SECONDS:
            flags.append(f"suspicious: reported DONE but used only {cpu:.1f} s of CPU in {run or 0:.0f} s "
                         f"({0 if mem is None else mem:.0f} MB); the program probably never ran. "
                         f"Check the logs and the output.")
        if any("Traceback (most recent call last)" in l for l in err_tail):
            flags.append("the error log contains a Python traceback")
    return {
        "job_id": row["jobid"], "job_name": name, "status": status,
        "exit_code": None if row["exit_code"] in ("-", "") else row["exit_code"],
        "run_time_s": run, "cpu_time_s": cpu, "max_memory_mb": None if mem is None else round(mem),
        "depends_on": None if row["dependency"] in ("-", "") else row["dependency"],
        "output_log": row["output_file"], "error_log": row["error_file"],
        "output_tail": out_tail, "error_tail": err_tail, "flags": flags,
    }


def _read_log(path: Optional[str], limit: int = 2 * 1024 * 1024) -> str:
    if not path or path == "-" or not os.path.isfile(path):
        return ""
    try:
        with open(path, "rb") as handle:
            return handle.read(limit).decode("utf-8", "replace")
    except OSError:
        return ""


def _container_log_dir(path: str) -> Optional[str]:
    """'<x>.zarr/output' for a log inside a container's own output folder, else None.

    Pyramid level jobs log there; the shared 'output' folder next to a container is not
    specific to one conversion, so it is not used to link jobs.
    """
    folder = os.path.dirname(path or "")
    parent = os.path.basename(os.path.dirname(folder))
    return folder if os.path.basename(folder) == "output" and parent.endswith((".zarr", ".n5")) else None


def related_jobs(job_id: str, everything: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """All jobs that belong to the chain started by job_id, found until no new job turns up.

    A job belongs to the chain if it waits for a chain job (LSF dependency), if a chain
    job's stdout says it submitted it ("Coordinator job submitted: N"; the pyramid
    coordinator starts the level jobs itself, without a dependency), or if it logs into
    the same container-local output folder as a chain job.
    """
    by_id = {r["jobid"]: r for r in everything}
    chain, seen, frontier = [], {job_id}, [job_id]
    while frontier:
        current = frontier.pop()
        row = by_id.get(current) or next((r for r in everything if r["jobid"] == current), None)
        found = [r for r in everything
                 if r["jobid"] not in seen and re.search(rf"\b{re.escape(current)}\b", r["dependency"])]
        if row is not None:
            for sid in re.findall(r"[Cc]oordinator job submitted:\s*(\d+)", _read_log(row["output_file"])):
                if sid in by_id and sid not in seen:
                    found.append(by_id[sid])
            local = _container_log_dir(row["output_file"])
            if local:
                found += [r for r in everything if r["jobid"] not in seen
                          and os.path.dirname(r["output_file"]) == local]
        for r in found:
            if r["jobid"] not in seen:
                seen.add(r["jobid"])
                chain.append(r)
                frontier.append(r["jobid"])
    return chain


def chain_state(jobs: List[Dict[str, Any]]) -> str:
    """'running' while anything is queued or running, else 'failed' / 'suspicious' / 'done'."""
    if any(j["status"] in ACTIVE for j in jobs):
        return "running"
    if any(j["status"] == "EXIT" for j in jobs):
        return "failed"
    if any(j["flags"] for j in jobs):
        return "suspicious"
    if any(j["status"] == "NOT_FOUND" for j in jobs):
        return "unknown"
    return "done"


def job_report(job_ids: List[str], follow_dependents: bool = True, log_lines: int = 15,
               runner: Optional[Callable] = None) -> Dict[str, Any]:
    """Report on jobs (and, by default, on the jobs that wait for them)."""
    runner = runner or subprocess.run
    requested = _run_bjobs(job_ids, runner)
    by_id = {r["jobid"]: r for r in requested}
    everything = _run_bjobs([], runner) if follow_dependents else []
    reports = []
    for job_id in job_ids:
        row = by_id.get(job_id)
        if row is None:
            reports.append({"job_id": job_id, "status": "NOT_FOUND", "flags": [],
                            "message": "LSF has no record of this job (too old, or the id is wrong)"})
            continue
        chain = [describe(row, log_lines)]
        if follow_dependents:
            chain += [describe(r, log_lines) for r in related_jobs(job_id, everything)]
        state = chain_state(chain)
        entry = chain[0]
        entry["chain"] = {"state": state, "jobs": [{"job_id": j["job_id"], "job_name": j["job_name"],
                                                    "status": j["status"], "flags": j["flags"]} for j in chain]}
        if len(chain) > 1:
            entry["dependents"] = chain[1:]
        reports.append(entry)
    return {"jobs": reports}
