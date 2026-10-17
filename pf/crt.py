"""Solve certificates: the status, primal value, dual bound and gap read back from the solver.

A certificate is built only from fields the solver returned. A requested tolerance is never
reported as a result: HiGHS stops at mip_rel_gap = 1e-4 unless told otherwise, and SCIP at
limits/gap = 0. A certificate whose dual bound lies past its primal value by more than the
tolerance is a defect and is refused.

Statuses: opt, limit, infeasible, unbounded, numerical.
"""

import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

# Legend
# st   : status, one of ST
# sn   : objective sense, "min" or "max"
# f    : primal value read back (objective at the incumbent); nan when there is none
# bd   : dual bound read back; -inf (min) or +inf (max) when the solver returned none
# g    : achieved relative gap |f - bd| / max(|f|, ga); inf when either side is missing
# nd   : branch-and-bound nodes; t : wall seconds
# rt   : root dual bound (SCIP), when read
# ps   : presolve setting of the solve that produced the certificate
# tol  : tolerance of the crossed-bound check, relative to max(1, |f|)
# ga   : floor of the gap denominator
# rg   : achieved gap read from mip_gap or SCIP getGap; None for LP dual marginals
# c    : objective vector; A, lo, hi : rows; lb, ub : column bounds; it : integrality
# tl   : time limit (s); gp : requested relative gap

ST = ("opt", "limit", "infeasible", "unbounded", "numerical")
TOL = 1e-7
GA = 1e-10


def gap(f, bd, ga=GA):
    """Relative gap |f - bd| / max(|f|, ga)."""
    if not (np.isfinite(f) and np.isfinite(bd)):
        return np.inf
    return abs(f - bd) / max(abs(f), ga)


def crossed(sn, f, bd, tol=TOL):
    """True when the dual bound lies past the primal value by more than tol (relative)."""
    if not (np.isfinite(f) and np.isfinite(bd)):
        return False
    s = tol * max(1.0, abs(f))
    return bd > f + s if sn == "min" else bd < f - s


def C(st, sn, f=np.nan, bd=None, nd=None, t=None, tol=TOL, rg=None, **k):
    """A certificate; refuses an unknown status or sense, and crossed bounds."""
    if st not in ST:
        raise ValueError(f"status {st!r} not in {ST}")
    if sn not in ("min", "max"):
        raise ValueError(f"sense {sn!r}")
    f = np.nan if f is None else float(f)
    if bd is None:
        bd = -np.inf if sn == "min" else np.inf
    bd = float(bd)
    if crossed(sn, f, bd, tol):
        raise ValueError(f"crossed certificate: sense {sn}, primal {f!r}, dual bound {bd!r}")
    if st == "opt" and not np.isfinite(f):
        raise ValueError("status opt without a primal value")
    if st == "opt" and not np.isfinite(bd):
        raise ValueError("status opt without a finite dual bound")
    return dict(st=st, sn=sn, f=f, bd=bd, g=gap(f, bd), rg=rg, nd=nd, t=t, **k)


_MS = {0: "opt", 1: "limit", 2: "infeasible", 3: "unbounded", 4: "numerical"}


def limits(tl, gp=0.0):
    """Reject nonfinite or negative time and gap limits."""
    if tl is not None and (not np.isfinite(tl) or tl < 0):
        raise ValueError("tl must be finite and nonnegative")
    if not np.isfinite(gp) or gp < 0:
        raise ValueError("gp must be finite and nonnegative")


def lp(c, A, lo, hi, lb, ub, tl, ps):
    """Solve an LP and evaluate the dual objective from HiGHS marginals."""
    eq = lo == hi
    il, iu = ~eq & np.isfinite(lo), ~eq & np.isfinite(hi)
    D = sparse.vstack([A[iu], -A[il]], format="csr")
    b = np.r_[hi[iu], -lo[il]]
    o = {"presolve": ps}
    if tl is not None:
        o["time_limit"] = float(tl)
    r = linprog(c, A_ub=D, b_ub=b, A_eq=A[eq], b_eq=lo[eq],
                bounds=np.column_stack([lb, ub]), method="highs", options=o)
    bd = None
    if r.status == 0:
        y, v = r.eqlin.marginals, r.ineqlin.marginals
        l, u = r.lower.marginals, r.upper.marginals
        ds = np.asarray(A[eq].T @ y + D.T @ v).ravel() + l + u
        er = np.max(np.abs(ds - c))
        if (er > TOL * max(1.0, np.abs(c).max()) or np.any(v > TOL)
                or np.any(l < -TOL) or np.any(u > TOL)):
            raise ValueError("LP solver marginals are not dual feasible")
        fl, fu = np.isfinite(lb), np.isfinite(ub)
        if np.any(np.abs(l[~fl]) > TOL) or np.any(np.abs(u[~fu]) > TOL):
            raise ValueError("LP solver dual multiplier on an infinite bound")
        bd = float(lo[eq] @ y + b @ v + lb[fl] @ l[fl] + ub[fu] @ u[fu])
    return r, bd


def mi(c, A, lo, hi, lb, ub, it, tl=None, gp=0.0, ps=True):
    """Minimise c'x over the rows with scipy.optimize.milp (HiGHS); returns (x, certificate).

    A reported infeasibility is read back from a second solve with presolve off: HiGHS presolve
    declared feasible programs infeasible in
    Electricity-Clearing-and-Pricing.
    """
    limits(tl, gp)
    if tl == 0:
        return None, C("limit", "min", t=0.0, ps=ps, bs=None, msg="time limit before solve")
    c, A = np.asarray(c, float), sparse.csr_matrix(A)
    lo, hi, lb, ub, it = [np.asarray(x) for x in (lo, hi, lb, ub, it)]
    if not np.isfinite(c).all() or not np.isfinite(A.data).all():
        raise ValueError("objective and row coefficients must be finite")
    if any(np.isnan(x).any() for x in (lo, hi, lb, ub, it)):
        raise ValueError("row bounds, column bounds and integrality must not contain NaN")
    o = {"mip_rel_gap": gp, "presolve": ps}
    if tl is not None:
        o["time_limit"] = float(tl)
    t0 = time.perf_counter()
    if np.any(it):
        r = milp(c, integrality=it, bounds=Bounds(lb, ub),
                 constraints=LinearConstraint(A, lo, hi), options=o)
        bd, bs = getattr(r, "mip_dual_bound", None), "mip_dual_bound"
    else:
        r, bd = lp(c, A, lo, hi, lb, ub, tl, ps)
        bs = "linprog dual marginals"
    t = time.perf_counter() - t0
    st = _MS.get(r.status, "numerical")
    if st == "infeasible" and ps:
        rem = None if tl is None else max(0.0, tl - t)
        x, c2 = mi(c, A, lo, hi, lb, ub, it, tl=rem, gp=gp, ps=False)
        c2["t"] += t
        c2["first"] = "infeasible"
        return x, c2
    f = r.fun if r.x is not None else None
    nd = getattr(r, "mip_node_count", None)
    return r.x, C(st, "min", f, bd, nd=nd, t=t, ps=ps, bs=bs, msg=r.message,
                  rg=getattr(r, "mip_gap", None))


def un(c, sg):
    """Certificate c of an objective scaled by sg > 0, in the original units (gap unchanged)."""
    if not np.isfinite(sg) or sg <= 0:
        raise ValueError("objective scale must be finite and positive")
    o = dict(c, f=c["f"] / sg, bd=c["bd"] / sg, sg=sg)
    o["g"] = gap(o["f"], o["bd"])
    if c.get("rt") is not None:
        o["rt"] = c["rt"] / sg
    return o


_SS = {"optimal": "opt", "timelimit": "limit", "nodelimit": "limit", "totalnodelimit": "limit",
       "stallnodelimit": "limit", "gaplimit": "limit", "memlimit": "limit", "sollimit": "limit",
       "bestsollimit": "limit", "userinterrupt": "limit", "infeasible": "infeasible",
       "unbounded": "unbounded", "inforunbd": "numerical"}
# An ambiguous status does not certify infeasibility; the raw status is retained.


def sc(m, sn, t=None, tol=TOL):
    """Certificate of a solved pyscipopt model: getStatus, getPrimalbound, getDualbound."""
    s = m.getStatus()
    st = _SS.get(s, "numerical")
    has = m.getNSols() > 0
    f = m.getPrimalbound() if has else None
    bd = m.getDualbound()
    inf = m.infinity()
    if abs(bd) >= inf:
        bd = None
    rt = m.getDualboundRoot()
    rt = None if abs(rt) >= inf else float(rt)
    return C(st, sn, f, bd, nd=m.getNNodes(), t=t if t is not None else m.getSolvingTime(),
             tol=tol, scip=s, rt=rt, rg=float(m.getGap()), bs="SCIP getDualbound")


def so(m, sn, tl=None):
    """Optimize a SCIP model, rechecking infeasibility without presolve within the deadline."""
    from pyscipopt import SCIP_PARAMSETTING
    limits(tl)
    if tl == 0:
        return C("limit", sn, t=0.0, ps=True, msg="time limit before solve")
    if tl is not None:
        m.setParam("limits/time", float(tl))
    t0 = time.perf_counter()
    m.optimize()
    first = str(m.getStatus())
    ps = True
    if first in ("infeasible", "inforunbd"):
        rem = None if tl is None else max(0.0, tl - (time.perf_counter() - t0))
        if rem == 0:
            return C("limit", sn, t=time.perf_counter() - t0, ps=True, first=first,
                     msg="no time for infeasibility recheck")
        m.freeTransform()
        m.setPresolve(SCIP_PARAMSETTING.OFF)
        if rem is not None:
            m.setParam("limits/time", rem)
        m.optimize()
        ps = False
    c = sc(m, sn, t=time.perf_counter() - t0)
    return dict(c, ps=ps, first=first)
