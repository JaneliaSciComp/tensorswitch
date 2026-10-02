"""
check_job_status reporting: tells a real success from a silent no-op, shows log tails,
and reports a chain of dependent jobs as one. bjobs is replaced by canned output.
"""

import json
import os
import subprocess

import pytest

from tensorswitch_v2.utils import job_status as js

SEP = "-" * 60
LSF_TAIL = f"\n{SEP}\nSender: LSF System <lsfadmin@h01>\nSubject: Job 1: <x> Done\n\nResource usage summary:\n  CPU time : 1 sec.\n"


def row(jobid, name="tsv2_tif_to_zarr3_x", stat="DONE", exit_code="-", run="120 second(s)", cpu="90.0 second(s)",
        mem="13 Gbytes", out="-", err="-", dep="-"):
    return "|".join([str(jobid), name, stat, exit_code, run, cpu, mem, out, err, dep])


def runner_for(rows):
    """A fake subprocess.run: bjobs with ids lists those, without ids lists all."""
    def fake(cmd, **kw):
        ids = [a for a in cmd[cmd.index("-o") + 2:] if a.isdigit()]
        picked = [r for r in rows if not ids or r.split("|")[0] in ids]
        return subprocess.CompletedProcess(cmd, 0, stdout="\n".join(picked) + "\n", stderr="")
    return fake


def report(rows, ids, **kw):
    return js.job_report(ids, runner=runner_for(rows), **kw)["jobs"]


class TestParsing:
    def test_units(self):
        assert js._seconds("123 second(s)") == 123 and js._seconds("-") is None
        assert js._megabytes("13 Gbytes") == 13 * 1024 and js._megabytes("21 Mbytes") == 21
        assert js._megabytes("-") is None


class TestFlags:
    def test_a_good_done_job_has_no_flags(self):
        job = report([row(1)], ["1"])[0]
        assert job["status"] == "DONE" and job["flags"] == [] and job["chain"]["state"] == "done"

    def test_the_silent_no_op_is_flagged_even_though_it_says_done(self):
        # real numbers from the sg-quoting incident: DONE, 8 s, 0.1 s CPU, 21 MB
        job = report([row(1, run="8 second(s)", cpu="0.1 second(s)", mem="21 Mbytes")], ["1"])[0]
        assert job["status"] == "DONE"
        assert any("suspicious" in f and "never ran" in f for f in job["flags"])
        assert job["chain"]["state"] == "suspicious"

    def test_small_real_jobs_are_not_flagged(self):
        # real numbers: the s3 level job of the ExPID108 pyramid (correct output) and its coordinator
        level = report([row(1, name="tsv2_ds_s3_raw", run="10 second(s)", cpu="2.0 second(s)", mem="32 Mbytes")], ["1"])[0]
        coordinator = report([row(2, name="tsv2_pyramid_coordinator_x.zarr", run="13 second(s)", cpu="1.0 second(s)",
                                  mem="86 Mbytes")], ["2"])[0]
        assert level["flags"] == [] and coordinator["flags"] == []

    def test_low_memory_alone_is_not_suspicious(self):
        assert report([row(1, cpu="30.0 second(s)", mem="21 Mbytes")], ["1"])[0]["flags"] == []

    def test_non_tensorswitch_jobs_are_not_judged(self):
        assert report([row(1, name="my_own_job", cpu="0.1 second(s)")], ["1"])[0]["flags"] == []

    def test_failed_job(self):
        job = report([row(1, stat="EXIT", exit_code="2")], ["1"])[0]
        assert job["chain"]["state"] == "failed" and "exit code 2" in job["flags"][0]

    def test_unknown_job(self):
        job = report([], ["99"])[0]
        assert job["status"] == "NOT_FOUND" and "no record" in job["message"]

    def test_traceback_in_the_error_log_is_flagged(self, temp_dir):
        err = os.path.join(temp_dir, "err.log")
        open(err, "w").write("Traceback (most recent call last):\n  File x\nValueError: boom\n")
        job = report([row(1, err=err)], ["1"])[0]
        assert any("traceback" in f.lower() for f in job["flags"])


class TestLogs:
    def test_stdout_tail_drops_the_lsf_summary(self, temp_dir):
        out = os.path.join(temp_dir, "out.log")
        open(out, "w").write("converting\nwrote 12 chunks\n" + LSF_TAIL)
        assert report([row(1, out=out)], ["1"])[0]["output_tail"] == ["converting", "wrote 12 chunks"]

    def test_log_with_only_the_lsf_block_is_empty(self, temp_dir):
        out = os.path.join(temp_dir, "out.log")
        open(out, "w").write(LSF_TAIL.lstrip("\n"))
        assert report([row(1, out=out)], ["1"])[0]["output_tail"] == []

    def test_only_the_last_lines_are_returned(self, temp_dir):
        err = os.path.join(temp_dir, "err.log")
        open(err, "w").write("\n".join(f"line {i}" for i in range(100)))
        tail = report([row(1, err=err)], ["1"], log_lines=3)[0]["error_tail"]
        assert tail == ["line 97", "line 98", "line 99"]

    def test_missing_log_files_are_fine(self):
        job = report([row(1, out="/nope/out.log", err="-")], ["1"])[0]
        assert job["output_tail"] == [] and job["error_tail"] == []


class TestChains:
    CHAIN = [
        row(10, name="tsv2_nd2_to_zarr3_x"),
        row(11, name="tsv2_pyramid_coordinator_x.zarr", mem="86 Mbytes", cpu="1.0 second(s)", dep="done(10)"),
        row(12, name="tsv2_pyramid_x_raw", dep="done(11)"),
        row(13, name="tsv2_ds_s1_raw", stat="RUN", dep="done(12)"),
        row(20, name="unrelated", dep="done(5)"),
    ]

    def test_dependents_are_followed_transitively(self):
        job = report(self.CHAIN, ["10"])[0]
        assert [j["job_id"] for j in job["chain"]["jobs"]] == ["10", "11", "12", "13"]
        assert "unrelated" not in json.dumps(job)

    def test_chain_is_running_until_the_last_job_ends(self):
        assert report(self.CHAIN, ["10"])[0]["chain"]["state"] == "running"
        done = [r.replace("|RUN|", "|DONE|") for r in self.CHAIN]
        assert report(done, ["10"])[0]["chain"]["state"] == "done"

    def test_failure_anywhere_in_the_chain_fails_it(self):
        rows = [r.replace("|RUN|", "|EXIT|") for r in self.CHAIN]
        assert report(rows, ["10"])[0]["chain"]["state"] == "failed"

    def test_a_flag_on_a_later_job_marks_the_chain_suspicious(self):
        rows = [r.replace("|RUN|", "|DONE|") for r in self.CHAIN]
        rows[2] = row(12, name="tsv2_pyramid_x_raw", mem="20 Mbytes", cpu="0.1 second(s)", dep="done(11)")
        assert report(rows, ["10"])[0]["chain"]["state"] == "suspicious"

    def test_following_can_be_turned_off(self):
        job = report(self.CHAIN, ["10"], follow_dependents=False)[0]
        assert len(job["chain"]["jobs"]) == 1 and "dependents" not in job

    def test_a_job_id_is_not_matched_inside_a_longer_id(self):
        rows = [row(1), row(2, dep="done(1234)")]
        assert [j["job_id"] for j in report(rows, ["1"])[0]["chain"]["jobs"]] == ["1"]


class TestPyramidChains:
    """The pyramid coordinator starts the level jobs itself: no LSF dependency links them."""

    @pytest.fixture
    def chain(self, temp_dir):
        shared = os.path.join(temp_dir, "output")
        local = os.path.join(temp_dir, "vol.zarr", "output")
        os.makedirs(shared)
        os.makedirs(local)
        coord_log = os.path.join(shared, "out_coord.log")
        open(coord_log, "w").write("PYRAMID PLAN\n  Coordinator job submitted: 12\n" + LSF_TAIL)
        rows = [
            row(10, name="tsv2_nd2_to_zarr3_x", out=os.path.join(shared, "out_conv.log")),
            row(11, name="tsv2_pyramid_coordinator_vol.zarr", mem="86 Mbytes", cpu="1.0 second(s)",
                dep="done(10)", out=coord_log),
            row(12, name="tsv2_pyramid_vol_raw", out=os.path.join(local, "out_12.log")),
            row(13, name="tsv2_ds_s1_raw", out=os.path.join(local, "out_13.log")),
            row(14, name="tsv2_ds_s2_raw", stat="RUN", out=os.path.join(local, "out_14.log")),
            row(30, name="other_conversion", out=os.path.join(shared, "out_other.log")),
        ]
        return rows

    def test_level_jobs_are_found_through_the_coordinator_log(self, chain):
        ids = [j["job_id"] for j in report(chain, ["10"])[0]["chain"]["jobs"]]
        assert ids == ["10", "11", "12", "13", "14"] or sorted(ids) == ["10", "11", "12", "13", "14"]

    def test_unrelated_jobs_sharing_the_output_folder_are_not_pulled_in(self, chain):
        assert "30" not in [j["job_id"] for j in report(chain, ["10"])[0]["chain"]["jobs"]]

    def test_chain_is_running_while_a_level_job_runs(self, chain):
        assert report(chain, ["10"])[0]["chain"]["state"] == "running"

    def test_chain_finishes_when_the_last_level_job_does(self, chain):
        done = [r.replace("|RUN|", "|DONE|") for r in chain]
        assert report(done, ["10"])[0]["chain"]["state"] == "done"

    def test_a_failed_level_job_fails_the_chain(self, chain):
        failed = [r.replace("|RUN|", "|EXIT|") for r in chain]
        assert report(failed, ["10"])[0]["chain"]["state"] == "failed"


class TestMcpTool:
    @pytest.fixture
    def tool(self, monkeypatch):
        pytest.importorskip("mcp")
        from tensorswitch_v2 import mcp_server

        def call(rows, ids, **kw):
            monkeypatch.setattr(subprocess, "run", runner_for(rows))
            return json.loads(mcp_server.check_job_status(ids, **kw))
        return call

    def test_single_job_keeps_the_old_keys(self, tool):
        result = tool([row(1)], "1")
        assert {"job_id", "status", "job_name"} <= set(result) and result["status"] == "DONE"

    def test_several_ids(self, tool):
        result = tool([row(1), row(2)], "1, 2")
        assert [j["job_id"] for j in result["jobs"]] == ["1", "2"]

    def test_no_bjobs(self, monkeypatch):
        pytest.importorskip("mcp")
        from tensorswitch_v2 import mcp_server

        def missing(*a, **k):
            raise FileNotFoundError("bjobs")
        monkeypatch.setattr(subprocess, "run", missing)
        assert json.loads(mcp_server.check_job_status("1"))["error"] == "bjobs_not_found"
