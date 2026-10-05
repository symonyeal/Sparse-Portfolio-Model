"""Stored results must retain their inputs, code identity and complete JSON record."""

import json
import os
import subprocess

import numpy as np
import pytest

import run_bench
from pf import bd

# Legend
# repo : isolated Git fixture; ip : expected raw input hashes; pa : case parameters


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pf").mkdir()
    (tmp_path / "pf" / "model.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    for args in (("init",), ("add", "."),
                 ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                  "commit", "-m", "Fixture")):
        subprocess.run(["git", "-C", str(tmp_path), *args], capture_output=True, check=True)
    return tmp_path


def saved(repo, **code_changes):
    code = bd.identity(repo) | code_changes
    out = repo / "docs" / "results"
    b = bd.bundle("case", {"K": 2}, {"value": 1}, {"data": "abc"}, [], code)
    bd.write(out, b)
    return out, code


def test_changed_inputs_invalidate_resume(repo):
    out, code = saved(repo)
    assert bd.valid(out, "case", {"K": 2}, code["source"], ip={"data": "abc"}, root=repo)
    assert bd.valid(out, "case", {"K": 2}, code["source"],
                    ip={"data": "changed"}, root=repo) is None
    assert bd.valid(out, "case", {"K": 2}, code["source"], root=repo) is None


def test_code_config_and_parameters_invalidate_resume(repo):
    out, _ = saved(repo)
    assert bd.valid(out, "case", {"K": 3}, ip={"data": "abc"}, root=repo) is None
    (repo / "requirements.txt").write_text("numpy\nscipy\n", encoding="utf-8")
    assert bd.valid(out, "case", {"K": 2}, ip={"data": "abc"}, root=repo) is None


@pytest.mark.parametrize("code", [{"dirty": True}, {"commit": "0" * 40}, {"commit": "HEAD"}])
def test_dirty_or_nonexistent_revision_cannot_be_admitted(repo, code):
    out, c = saved(repo, **code)
    assert bd.valid(out, "case", {"K": 2}, c["source"],
                    ip={"data": "abc"}, root=repo) is None


def test_git_failure_never_means_clean_revision(tmp_path):
    subprocess.run(["git", "-C", str(tmp_path), "init"], capture_output=True, check=True)
    c = bd.identity(tmp_path)
    assert c["dirty"] and c["commit"] is None


def test_damaged_bundle_is_rerunnable(repo):
    out, c = saved(repo)
    (out / "case.json").write_text('{"bundle":', encoding="utf-8")
    assert bd.valid(out, "case", {"K": 2}, c["source"],
                    ip={"data": "abc"}, root=repo) is None


def test_nonfinite_solver_fields_are_json_null(repo):
    out, c = saved(repo)
    b = bd.bundle("case", {}, {"f": np.nan, "bd": -np.inf, "g": np.inf}, {}, [], c)
    p = bd.write(out, b)

    def invalid(value):
        pytest.fail(f"nonstandard JSON constant {value}")

    r = json.loads(p.read_text(encoding="utf-8"), parse_constant=invalid)
    assert (r["f"], r["bd"], r["g"]) == (None, None, None)


def test_interrupted_replace_preserves_previous_bundle(repo, monkeypatch):
    out, c = saved(repo)
    before = (out / "case.json").read_bytes()

    def fail(*args):
        raise OSError("interrupted replacement")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="interrupted"):
        bd.write(out, bd.bundle("case", {}, {"value": 2}, {}, [], c))
    assert (out / "case.json").read_bytes() == before


def test_table_excludes_unvalidated_records(tmp_path, monkeypatch):
    out = tmp_path / "bundles"
    out.mkdir()
    (out / "foreign.json").write_text(json.dumps({"bundle": {
        "code": {"commit": "f" * 40, "source": "bad", "dirty": True},
        "host": {"node": "untrusted-host"}}}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_bench, "OUT", out)
    run_bench.table()
    report = (tmp_path / "docs/results/README.md").read_text(encoding="utf-8")
    assert "untrusted-host" not in report
    assert "Revisions: none" in report


@pytest.mark.parametrize("args", [["--time", "nan"], ["--time", "inf"], ["--time", "0"],
                                  ["--time", "-1"], ["--jobs", "0"], ["--jobs", "-2"]])
def test_invalid_cli_limits_are_rejected(args, monkeypatch):
    monkeypatch.setattr(run_bench, "table", lambda: None)
    with pytest.raises(SystemExit) as e:
        run_bench.main(args)
    assert e.value.code != 0


def test_table_admits_matching_inputs_and_formats_missing_incumbent(repo, monkeypatch):
    key = "g2-port1-lam0.25"
    pa = {"k": 1, "K": 10, "e": .01, "lam": .25, "tl": 60.0}
    ip = {"port1": run_bench.dat.sha256(run_bench.dat.ORLIB / "port1.txt")}
    c = {"st": "limit", "f": None, "bd": None, "it": 0}
    r = {"scip": c, "ecp": c, "scip_f": None, "ecp_f": None,
         "rel": None, "in_bracket": None, "pass": None}
    out = repo / "bundles"
    bd.write(out, bd.bundle(key, pa, r, ip, [], bd.identity(repo)))
    monkeypatch.chdir(repo)
    monkeypatch.setattr(bd, "ROOT", repo)
    monkeypatch.setattr(run_bench, "OUT", out)
    run_bench.table()
    p = repo / "docs/results/README.md"
    assert "limit n/a" in p.read_text(encoding="utf-8")
    b = bd.read(out / f"{key}.json")
    b["bundle"]["inputs"]["port1"] = "stale"
    bd.write(out, b)
    (out / "broken.json").write_text("{", encoding="utf-8")
    run_bench.table()
    assert key not in p.read_text(encoding="utf-8")


def test_resume_uses_current_input_hashes(repo, monkeypatch):
    key = "g1-port1"
    ip = {n: run_bench.dat.sha256(run_bench.dat.ORLIB / f"{n}.txt")
          for n in ("port1", "portef1")}
    out = repo / "bundles"
    b = bd.bundle(key, {"k": 1, "tl": 10.0}, {}, ip, [], bd.identity(repo))
    bd.write(out, b)
    monkeypatch.setattr(bd, "ROOT", repo)
    monkeypatch.setattr(run_bench, "OUT", out)
    monkeypatch.setattr(run_bench, "table", lambda: None)
    monkeypatch.setattr(bd, "identity", lambda: b["bundle"]["code"])
    ran = []
    monkeypatch.setattr(run_bench, "one", lambda a: (ran.append(a[0]) or a[0], 0.0))
    args = ["--run", "--missing", "--only", key, "--time", "10"]
    run_bench.main(args)
    assert ran == []
    b["bundle"]["inputs"]["port1"] = "stale"
    bd.write(out, b)
    run_bench.main(args)
    assert ran == [key]


@pytest.mark.parametrize("args", [["--only", "chang-port1", "--time", "60"],
                                  ["--only", "f4", "--time", "200"]])
def test_long_batch_requires_explicit_override(repo, monkeypatch, args):
    code = bd.identity(repo)
    monkeypatch.setattr(bd, "identity", lambda: code)
    monkeypatch.setattr(run_bench, "table", lambda: None)
    monkeypatch.setattr(run_bench, "OUT", repo / "bundles")
    monkeypatch.setattr(run_bench, "one", lambda a: (a[0], 0.0))
    with pytest.raises(SystemExit, match="30 minutes"):
        run_bench.main(["--run", *args])


def test_unknown_case_is_not_a_successful_empty_run(repo, monkeypatch):
    code = bd.identity(repo)
    monkeypatch.setattr(bd, "identity", lambda: code)
    with pytest.raises(SystemExit):
        run_bench.main(["--run", "--only", "typo"])
