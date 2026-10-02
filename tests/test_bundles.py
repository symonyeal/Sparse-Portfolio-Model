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
