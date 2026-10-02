"""X(K) rows, certificates and strict loaders."""

import zipfile

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, milp

from pf import crt, dat, rw

# Legend
# B : row block of X(K); x : a column vector [w, z]; p : temporary file path


def inrows(B, x, tol=1e-9):
    a = B["A"] @ x
    return bool(np.all(a >= B["lo"] - tol) and np.all(a <= B["hi"] + tol))


@pytest.mark.parametrize("sn", rw.SN)
def test_rows_match_definition(sn):
    N, K = 6, 3
    B = rw.X(N, K, 0.05, 0.6, sn)
    w = np.array([0.5, 0.3, 0.2, 0, 0, 0])
    z = np.array([1, 1, 1, 0, 0, 0])
    assert inrows(B, np.r_[w, z]) and rw.ok(w, z, K, 0.05, 0.6, sn) == []
    z2 = np.array([1, 1, 0, 0, 0, 0])                   # two names admitted, three held
    assert not inrows(B, np.r_[w, z2]) and rw.ok(w, z2, K, 0.05, 0.6, sn)
    w3 = np.array([0.7, 0.2, 0.1, 0, 0, 0])             # ceiling 0.6 violated
    assert not inrows(B, np.r_[w3, z]) and rw.ok(w3, z, K, 0.05, 0.6, sn)
    z4 = np.array([1, 1, 0, 0, 0, 0])
    w4 = np.array([0.5, 0.5, 0, 0, 0, 0])
    assert inrows(B, np.r_[w4, z4]) == (sn == "<=")      # fewer than K names only under <=
    assert (rw.ok(w4, z4, K, 0.05, 0.6, sn) == []) == (sn == "<=")


def test_exact_cardinality_admits_zero_weight_with_zero_floor():
    B = rw.X(4, 3, 0.0, 1.0, "==")
    w, z = np.array([1.0, 0, 0, 0]), np.array([1, 1, 1, 0])
    assert inrows(B, np.r_[w, z]) and rw.ok(w, z, 3, 0.0, 1.0, "==") == [] and rw.nz(w) == 1


@pytest.mark.parametrize("bad", [{"K": 0}, {"K": 7}, {"e": -0.1}, {"d": 1.2},
                                 {"e": 0.5, "d": 0.4}, {"sn": "<"}])
def test_rows_refuse_bad_parameters(bad):
    a = {"N": 6, "K": 3, "e": 0.0, "d": 1.0, "sn": "<="} | bad
    with pytest.raises(ValueError):
        rw.X(**a)


@pytest.mark.parametrize("sn", ["min", "max"])
def test_crossed_certificate_refused(sn):
    f, bd = (1.0, 1.1) if sn == "min" else (1.0, 0.9)
    with pytest.raises(ValueError, match="crossed"):
        crt.C("opt", sn, f, bd)
    c = crt.C("opt", sn, 1.0, 1.0 - 1e-9 if sn == "min" else 1.0 + 1e-9)
    assert c["g"] < 1e-8


def knap(n=60, seed=0):
    g = np.random.default_rng(seed)
    c = -g.integers(1, 100, n).astype(float)
    A = g.integers(1, 100, (20, n)).astype(float)
    return c, A, np.full(20, -np.inf), A.sum(1) / 3, np.zeros(n), np.ones(n), np.ones(n)


def test_time_limited_solve_is_limit_with_its_bounds():
    _, c = crt.mi(*knap(), tl=0.05)
    assert c["st"] == "limit"
    assert np.isfinite(c["f"]) and c["bd"] <= c["f"] and c["g"] > 0
    _, c = crt.mi(*knap(), tl=1e-4)
    assert c["st"] == "limit" and np.isnan(c["f"]) and c["bd"] == -np.inf


def test_infeasible_read_back_with_presolve_off():
    x, c = crt.mi([1, 1], np.array([[1, 1]]), [3], [3], [0, 0], [1, 1], [1, 1])
    assert c["st"] == "infeasible" and c["ps"] is False and x is None


def test_milp_reference_read_back():
    r = milp([1, 1], integrality=[1, 1], bounds=Bounds(0, 5),
             constraints=LinearConstraint([[1, 1]], 3, np.inf), options={"mip_rel_gap": 0})
    _, c = crt.mi([1, 1], np.array([[1, 1]]), [3], [np.inf], [0, 0], [5, 5], [1, 1])
    assert c["st"] == "opt" and c["f"] == r.fun == c["bd"] and c["bs"] == "mip_dual_bound"


def test_orlib_pins_and_shapes():
    assert set(dat.pins()) == {f"{n}{k}.txt" for n, m in (("port", 5), ("portef", 5),
                                                         ("indtrack", 8)) for k in range(1, m + 1)}
    d = dat.port(1)
    assert d["N"] == 31 and np.allclose(d["S"], d["S"].T) and np.linalg.eigvalsh(d["S"])[0] > 0
    assert dat.portef(1)["ef"].shape == (2000, 2)
    it = dat.indtrack(1)
    assert it["V"].shape == (31, 291) and it["I"].shape == (291,)


@pytest.mark.parametrize("k", [7, 8])
def test_trailing_indtrack_series_refused(k):
    with pytest.raises(dat.DataError, match="after the"):
        dat.indtrack(k)
    d = dat.indtrack(k, tail=True)
    assert d["drop"] == d["T"] + 1


def port_text(N=3, drop=None, extra="", rep=False):
    L = [str(N)] + [" .01 .05" for _ in range(N)]
    P = [(i, j) for i in range(1, N + 1) for j in range(i, N + 1)]
    if rep:
        P[-1] = P[-2]
    L += [f"{i} {j} {1.0 if i == j else 0.2}" for i, j in P if (i, j) != drop]
    return "\n".join(L) + "\n" + extra


@pytest.mark.parametrize("txt,msg", [(port_text(drop=(1, 2)), "tokens"),
                                     (port_text(extra="9 9 0.1"), "tokens"),
                                     (port_text(rep=True), "repeated"),
                                     (port_text().replace(" .05", " -.05", 1), "nonpositive")])
def test_malformed_port_refused(tmp_path, txt, msg):
    p = tmp_path / "port.txt"
    p.write_text(txt)
    with pytest.raises(dat.DataError, match=msg):
        dat.port(0, path=p)


def test_malformed_indtrack_and_portef_refused(tmp_path):
    p = tmp_path / "it.txt"
    p.write_text("2 2\n1 2 3\n1 2 3\n1 2\n")
    with pytest.raises(dat.DataError, match="values"):
        dat.indtrack(0, path=p)
    q = tmp_path / "ef.txt"
    q.write_text(".01 .001\n.02\n")
    with pytest.raises(dat.DataError, match="odd"):
        dat.portef(0, path=q)


CR = ["PERMNO,PERMCO,date,SHRCD,EXCHCD,PRC,SHROUT,RET,DLSTCD,DLRET",
      "10001,7953,19860131,11,3,-2.5,1000,C,,",
      "10001,7953,19860228,11,3,3.0,1000,0.2,,",
      "10001,7953,19860331,11,3,2.4,1000,-0.2,560,",
      "10002,7954,19860131,73,1,10,50,0.01,,"]


def test_crsp_missing_stays_missing_and_extra_field_refused(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("\n".join(CR) + "\n")
    d = dat.crsp(p)
    assert d["dropped"] == 1 and d["PERMNO"].tolist() == [10001] * 3
    assert np.isnan(d["RET"][0]) and d["RETc"][0] == "C" and np.isnan(d["DLRET"][2])
    q = tmp_path / "x.csv"
    q.write_text("\n".join([CR[0] + ",VOL"] + [r + ",5" for r in CR[1:]]) + "\n")
    with pytest.raises(dat.DataError, match="unread"):
        dat.crsp(q)


def test_french_rf_parser(tmp_path):
    s = ("header\n\n,Mkt-RF,SMB,HML,RF\n192607,   2.89,  -2.42,  -2.75,   0.22\n"
         "192608,   2.64,  -1.44,   4.13,   0.25\n\n Annual Factors: January-December\n"
         ",Mkt-RF,SMB,HML,RF\n  1927,  29.44,  -3.05,  -3.36,   3.12\n")
    p = tmp_path / "ff.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("F-F_Research_Data_Factors.csv", s)
    d = dat.rf(p)
    assert d["mo"].tolist() == [192607, 192608] and np.allclose(d["rf"], [0.0022, 0.0025])
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("F-F_Research_Data_Factors.csv", s.replace("192608", "192609"))
    with pytest.raises(dat.DataError, match="consecutive"):
        dat.rf(p)
