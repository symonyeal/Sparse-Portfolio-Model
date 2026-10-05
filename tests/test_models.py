"""Gates G1 and G2, F1 = F1p, and the independent checks of F2-F5."""

from itertools import pairwise

import numpy as np
import pytest

from pf import cv, dat, ecp, it, mad, mv, rw, sk

# Legend
# mu, S : port1 moments; ef : portef1 frontier; r, Ri : indtrack1 log returns


@pytest.fixture(scope="module")
def p1():
    d = dat.port(1)
    return d["mu"], d["S"]


@pytest.fixture(scope="module")
def r1():
    d = dat.indtrack(1)
    return dat.lr(d["V"]).T, dat.lr(d["I"])


@pytest.fixture(scope="module")
def g1_rectangles(p1):
    """Certified variances at each printed mean and its rounding-interval endpoints."""
    mu, S = p1
    ef = dat.portef(1)["ef"]
    F0, kt, frontier = np.ones(mu.size), [], []
    half = 5e-11
    for r, v in ef:
        lo, hi = max(r - half, mu.min()), min(r + half, mu.max())
        assert lo <= hi, f"printed mean {r} has no feasible rounding interval"
        values = []
        for mean in (lo, np.clip(r, lo, hi), hi):
            q = mv.qp(mu, S, mean, F0, it=5 * mu.size)
            F0 = q["w"] + (q["w"] > 0)
            values.append(q["f"])
            kt.append(max(q["kt"].values()))
        frontier.append(values)
    return ef, np.array(frontier), kt


def test_g1_frontier_certificates_at_printed_means(g1_rectangles):
    ef, frontier, kt = g1_rectangles
    assert max(kt) < 1e-12
    assert np.isfinite(frontier[:, 1] - ef[:, 1]).all()


@pytest.mark.xfail(strict=True, reason="Published portef1: 142 of 2000 printed rounding "
                   "rectangles miss the certified frontier (maximum separation 8.96e-11)")
def test_g1_printed_rectangles_intersect_frontier(g1_rectangles):
    ef, frontier, kt = g1_rectangles
    assert max(kt) < 1e-12
    half = 5e-11
    matched = ((frontier.max(axis=1) >= ef[:, 1] - half)
               & (frontier.min(axis=1) <= ef[:, 1] + half))
    assert matched.all(), np.flatnonzero(~matched) + 1


def test_g1_scip_route_at_K_equal_N(p1):
    mu, S = p1
    ef = dat.portef(1)["ef"]
    for i in (0, 999, 1999):
        q = mv.qp(mu, S, ef[i, 0], np.ones(mu.size), it=200)
        s = mv.f1(mu, S, mu.size, 1.0, rho=ef[i, 0], rs="==")
        assert s["c"]["st"] == "opt" and s["c"]["bd"] <= q["f"] * (1 + 1e-12)
        assert abs(s["f"] - q["f"]) <= 1e-7 * q["f"]


@pytest.mark.parametrize("sn", rw.SN)
@pytest.mark.parametrize("lam", [0.5, 1.0])
def test_f1_equals_f1p_and_lies_in_X(p1, lam, sn):
    mu, S = p1
    a = mv.f1(mu, S, 10, lam, e=0.01, sn=sn)
    b = mv.f1(mu, S, 10, lam, e=0.01, sn=sn, p=True)
    assert a["c"]["st"] == b["c"]["st"] == "opt"
    assert abs(a["f"] - b["f"]) <= 1e-7 * max(abs(a["f"]), 1e-12)
    for x in (a, b):
        assert rw.ok(x["w"], x["z"], 10, 0.01, 1.0, sn) == []


@pytest.mark.parametrize("lam", [0.0, 0.25])
def test_g2_scip_and_ecp_agree_where_both_certify(p1, lam):
    mu, S = p1
    s = mv.f1(mu, S, 10, lam, e=0.01, sn="==")
    q = ecp.ecp(mu, S, 10, lam, e=0.01, sn="==", tl=60)
    assert s["c"]["st"] == q["c"]["st"] == "opt"
    assert abs(s["f"] - q["f"]) <= 1e-6 * abs(s["f"])
    assert rw.ok(q["w"], q["z"], 10, 0.01, 1.0, "==") == []


def test_g2_ecp_bracket_contains_scip_optimum(p1):
    mu, S = p1
    s = mv.f1(mu, S, 10, 1.0, e=0.01, sn="==")
    q = ecp.ecp(mu, S, 10, 1.0, e=0.01, sn="==", it=100)
    tol = 1e-9 * s["f"]
    assert q["c"]["bd"] - tol <= s["f"] <= q["c"]["f"] + tol
    lbs = [p[1] for p in q["pt"]]
    assert all(b >= a - 1e-15 for a, b in pairwise(lbs))           # monotone, their eq. (15)


@pytest.mark.parametrize("sn", rw.SN)
def test_f2_deviation_recomputed(r1, sn):
    r, _ = r1
    s = mad.mad(r, 10, rho=float(np.median(r.mean(axis=0))), sn=sn)
    assert s["c"]["st"] == "opt"
    assert abs(s["c"]["f"] - s["f"]) <= 1e-9 * s["f"]
    assert rw.ok(s["w"], s["z"], 10, sn=sn) == []
    assert r.mean(axis=0) @ s["w"] >= float(np.median(r.mean(axis=0))) - 1e-9


@pytest.mark.parametrize("b", [0.95, 0.9537])
def test_f3_cvar_identity(r1, b):
    r, _ = r1
    s = cv.cv(r, 10, b=b)
    assert s["c"]["st"] == "opt"
    assert abs(s["c"]["f"] - s["f"]) <= 1e-9 * abs(s["f"])
    assert rw.ok(s["w"], s["z"], 10) == []


def test_cvar_by_sorting_hand_case():
    r = np.array([[0.1], [-0.2], [0.05], [-0.4], [0.0]])
    assert np.isclose(cv.cvar(r, np.ones(1), 0.6), 0.3)            # mean of the 2 worst losses
    assert np.isclose(cv.cvar(r, np.ones(1), 0.7), (0.4 + 0.5 * 0.2) / 1.5)


def test_f4_stages_and_regression_identity(r1):
    r, Ri = r1
    tr = slice(0, 145)
    s = it.it(r[tr], Ri[tr], 10)
    assert s["c1"]["st"] == s["c2"]["st"] == "opt"
    a, b = it.ols(r[tr] @ s["w"], Ri[tr])
    assert np.isclose(a, s["a"], atol=1e-12) and np.isclose(b, s["b"], atol=1e-12)
    assert abs(s["a"]) <= 1e-9 + 1e-12 and abs(s["b"] - 1) <= s["c2"]["f"] + 1e-12
    assert rw.ok(s["w"], s["z"], 10, 0.01, 1.0, "==") == []
    assert np.isfinite(it.te(r[145:], Ri[145:], s["w"]))


def test_f5_dual_bound_against_grid(r1):
    r, _ = r1
    x = r[:36, :6]          # 60 scenarios ran past 120 s at limits/gap 0; 36 certify in 0.1 s
    s = sk.f5(x, 2, 3.0, tl=60)
    gb, _ = sk.grid(x, 2, 3.0, h=0.01)
    assert s["c"]["st"] == "opt"
    assert s["c"]["bd"] >= gb - 1e-12 and s["f"] >= gb - 1e-10
    assert rw.ok(s["w"], s["z"], 2) == []
    assert np.isclose(sk.kap(3.0), 12.0)


def test_shrinkage_against_appendix_loops():
    g = np.random.default_rng(3)
    y = g.standard_t(5, size=(60, 5)) @ g.normal(size=(5, 5)) * 0.03
    Sh, dl = mv.shrink(y)
    T, N = y.shape
    x = y - y.mean(0)
    S = x.T @ x / T
    s = np.sqrt(np.diag(S))
    rb = np.mean([S[i, j] / s[i] / s[j] for i in range(N) for j in range(N) if i < j])
    F = np.where(np.eye(N, dtype=bool), S, rb * np.outer(s, s))
    p = lambda i, j: np.mean((x[:, i] * x[:, j] - S[i, j]) ** 2)
    t = lambda a, i, j: np.mean((x[:, a] ** 2 - S[a, a]) * (x[:, i] * x[:, j] - S[i, j]))
    rho = sum(p(i, i) for i in range(N)) + sum(
        rb / 2 * (s[j] / s[i] * t(i, i, j) + s[i] / s[j] * t(j, i, j))
        for i in range(N) for j in range(N) if i != j)
    k = (sum(p(i, j) for i in range(N) for j in range(N)) - rho) / np.sum((F - S) ** 2)
    assert np.isclose(dl, max(0, min(k / T, 1)), rtol=1e-12)
    assert np.allclose(Sh, dl * F + (1 - dl) * S, atol=1e-16)
