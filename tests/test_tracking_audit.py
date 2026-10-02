"""F4 lexicographic priorities, evidence and invalid-input boundaries."""

import numpy as np
import pytest

from pf import crt, it, rw

# Legend
# Ri : four benchmark returns; r : two exact lines with different intercepts and slopes


@pytest.fixture
def lines():
    Ri = np.array([-0.03, -0.01, 0.01, 0.03])
    return np.c_[0.02 + Ri, 0.01 + 3 * Ri], Ri


def test_intercept_priority_and_both_objectives_are_recomputed(lines):
    r, Ri = lines
    s = it.it(r, Ri, 1)
    assert s["st"] == "opt"
    assert np.allclose(s["w"], [0, 1], atol=1e-9)
    assert np.isclose(s["f1"], 0.01, atol=1e-12)
    assert np.isclose(s["f2"], 2.0, atol=1e-12)
    assert np.isclose(s["f1"], s["c1"]["f"], atol=1e-12)
    assert np.isclose(s["f2"], s["c2"]["f"], atol=1e-12)
    assert rw.ok(s["w"], s["z"], 1, 0.01, 1.0, "==") == []


def test_stage_one_limit_cannot_certify_a_lexicographic_optimum(lines, monkeypatch):
    r, Ri = lines
    calls = []

    def limited(*args, **kwargs):
        calls.append(1)
        return np.array([1, 0, 1, 0, 0.02]), crt.C("limit", "min", 0.02, 0.0)

    monkeypatch.setattr(crt, "mi", limited)
    s = it.it(r, Ri, 1)
    assert s["st"] == "limit" and s["c2"] is None
    assert len(calls) == 1


def test_mismatched_readback_objective_is_rejected(lines, monkeypatch):
    r, Ri = lines

    def inconsistent(*args, **kwargs):
        return np.array([1, 0, 1, 0, 0.02]), crt.C("opt", "min", 0.2, 0.2)

    monkeypatch.setattr(crt, "mi", inconsistent)
    with pytest.raises(ValueError, match="objective"):
        it.it(r, Ri, 1)


@pytest.mark.parametrize("ea", [-1.0, np.nan, np.inf])
def test_invalid_stage_fix_tolerance_is_rejected(lines, ea):
    with pytest.raises(ValueError, match="ea"):
        it.it(*lines, 1, ea=ea)


@pytest.mark.parametrize("r,Ri", [([], []), ([[0.1], [np.nan]], [0.0, 0.1]),
                                    ([[0.1], [0.2]], [[0.0], [0.1]]),
                                    ([[0.1], [0.2]], [0.0]),
                                    ([[0.1], [0.2]], [0.0, np.inf])])
def test_malformed_or_nonfinite_regression_inputs_are_rejected(r, Ri):
    with pytest.raises(ValueError, match="returns|index|observations"):
        it.ols(r, Ri)
