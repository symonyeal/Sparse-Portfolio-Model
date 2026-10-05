"""Regression checks for invalid certificates, inputs, deadlines and grids.

Legend: r = scenarios; mu = mean; S = covariance; c = solve certificate.
"""

import numpy as np
import pytest

from pf import crt, cv, ecp, mad, mv, rw, sk


def test_nan_weight_is_not_a_feasible_portfolio():
    assert rw.ok([np.nan, 0.0], [1, 0], 1)


def test_optimal_certificate_requires_a_dual_bound():
    with pytest.raises(ValueError, match="dual bound"):
        crt.C("opt", "min", 1.0)


def test_lp_bound_is_obtained_from_dual_marginals():
    x, c = crt.mi([-1.0, 2.0], [[1.0, 1.0]], [1.0], [1.0], [0, 0], [2, 2], [0, 0])
    assert np.allclose(x, [1, 0]) and c["f"] == pytest.approx(-1)
    assert c["bd"] == pytest.approx(-1) and c["bs"] == "linprog dual marginals"


def test_highs_achieved_gap_is_kept():
    _, c = crt.mi([1.0], [[1.0]], [1.0], [1.0], [0], [1], [1])
    assert c["rg"] == 0.0


@pytest.mark.parametrize("fn", [mv.f1, ecp.ecp])
def test_convex_routes_reject_indefinite_covariance(fn):
    with pytest.raises(ValueError, match="semidefinite"):
        fn([0.01, 0.02], [[1.0, 2.0], [2.0, 1.0]], 1, 1.0, tl=1)


def test_perspective_accepts_singular_covariance():
    S = np.ones((2, 2)) * 0.01
    Q, pi = mv.split(S)
    assert np.all(pi >= 0) and np.allclose(Q + np.diag(pi), S)
    d = mv.f1([0.01, 0.02], S, 1, 1.0, p=True, tl=2)
    assert d["c"]["st"] == "opt" and d["f"] == pytest.approx(0.01, abs=1e-7)


@pytest.mark.parametrize("fn", [mv.f1, ecp.ecp])
def test_zero_deadline_starts_no_solve(fn):
    d = fn([0.01, 0.02], np.eye(2), 1, 1.0, tl=0)
    assert d["c"]["st"] == "limit" and d["w"] is None


def test_zero_deadline_does_not_return_an_unchecked_warm_start():
    d = mv.f1([0.01, 0.02], np.eye(2), 1, 1.0, tl=0, sol=([1, 0], [1, 0]))
    assert d["w"] is None and np.isnan(d["c"]["f"])


def test_ecp_retains_dual_bound_without_incumbent(monkeypatch):
    def mi(*a, **k):
        return None, crt.C("limit", "min", bd=0.25, rg=np.inf)
    monkeypatch.setattr(crt, "mi", mi)
    d = ecp.ecp([0.01, 0.02], np.eye(2), 1, 1.0)
    assert d["c"]["bd"] == 0.25 and d["w"] is None


@pytest.mark.parametrize("fn", [mad.mad, cv.cv])
def test_milp_models_reject_missing_returns(fn):
    with pytest.raises(ValueError, match="finite"):
        fn([[0.01, 0.02], [np.nan, 0.03]], 1)


def test_grid_refuses_a_step_that_does_not_tile_the_simplex():
    with pytest.raises(ValueError, match="reciprocal"):
        sk.grid([[0.01, 0.02], [0.03, 0.01]], 1, 2, h=0.3)


def test_grid_zero_floor_exact_cardinality_has_admitted_zero_names():
    r = np.array([[0.02, 0.01], [0.02, 0.01]])
    f, w = sk.grid(r, 2, 2, h=0.25, sn="==")
    assert f == pytest.approx(0.02) and np.allclose(w, [1, 0])


def test_scip_infeasibility_is_rechecked_with_presolve_off():
    d = mv.f1([0.01, 0.02], np.eye(2), 1, 1.0, d=0.2, tl=2)
    assert d["c"]["st"] == "infeasible" and d["c"]["ps"] is False


def test_scip_achieved_gap_is_kept():
    d = mv.f1([0.01, 0.02], np.eye(2), 1, 1.0, tl=2)
    assert d["c"]["st"] == "opt" and d["c"]["rg"] == pytest.approx(0.0)


def test_zero_covariance_does_not_make_nonfinite_scale():
    d = sk.f5([[0.01, 0.02], [0.01, 0.02]], 1, 2, tl=2)
    assert d["c"]["st"] == "opt" and d["f"] == pytest.approx(0.02)


def test_shrink_handles_one_asset_and_constant_series():
    S, dl = mv.shrink([[0.01], [0.03]])
    assert S.shape == (1, 1) and S[0, 0] == pytest.approx(0.0001) and dl == 0
    S, dl = mv.shrink([[0.01, 0.02], [0.03, 0.02]])
    assert np.isfinite(S).all() and np.allclose(S[1], 0) and 0 <= dl <= 1


def test_cvar_preserves_tiny_fractional_tail_atoms():
    b = np.nextafter(1.0, 0.0)
    assert cv.cvar([[-0.4], [-0.2]], [1.0], b) == pytest.approx(0.4)
