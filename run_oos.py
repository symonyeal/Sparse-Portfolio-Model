"""RQ2-RQ4: out-of-sample selection on CRSP under the pre-registered plan (docs/PLAN.md).

    python run_oos.py --crsp PATH [--rf PATH] [--only STRATEGY] [--K 10] [--M 120] [--H 120]
                      [--Nc 200] [--m 120] [--fill] [--time 60] [--missing]

Needs a licensed CRSP extract; without one it stops and P6 stays blocked. For each rebalance
date the selection sees P.upto(t) only; the holding is evaluated on the full panel. One bundle
per (strategy, K, M, m, Nc, H, fill) holds terminal wealths and held PERMCOs per date, never
row-level data. The set T of top wealth creators is computed afterwards by pf/wc.py and used
only in pf/ev.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from pf import bd, bt, cv, dat, mad, mv

# Legend
# P0 : panel; Pt : panel cut at t; U : eligible columns; sel : selection rule
# K, M, m, Nc, H : names, window, required returns, universe size, holding months
# tl : per-solve time limit; lam : Chang's weight for F1 (docs/PLAN.md)

OUT = Path("docs/results/oos")
LAM = 0.5
ST = ("b2", "b3", "b4", "f1", "f2", "f3")


def window(Pt, U, M):
    """Returns of the eligible columns with a complete M-month window (the optimizer's filter)."""
    y = Pt.R[-M:][:, U]
    k = np.isfinite(y).all(axis=0)
    return U[k], y[:, k]


def rule(s, K, M, tl, g):
    def sel(Pt, U):
        if s == "b2":
            return bt.b2(Pt, U)
        if s == "b3":
            return bt.b3(Pt, U, K)
        if s == "b4":
            return bt.b4(Pt, U, K, g)
        c, y = window(Pt, U, M)
        if s == "f1":
            S, _ = mv.shrink(y)
            x = mv.f1(y.mean(axis=0), S, K, LAM, tl=tl)
        elif s == "f2":
            x = mad.mad(y, K, tl=tl)
        else:
            x = cv.cv(y, K, b=0.95, tl=tl)
        if x["w"] is None:
            raise RuntimeError(f"{s}: no incumbent at {Pt.mo[-1]} ({x['c']['st']})")
        w = np.clip(x["w"], 0, None)
        return c, w / w.sum()
    return sel


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--crsp")
    a.add_argument("--rf")
    a.add_argument("--only", action="append", default=[])
    a.add_argument("--K", type=int, default=10)
    a.add_argument("--M", type=int, default=120)
    a.add_argument("--m", type=int, default=120)
    a.add_argument("--Nc", type=int, default=200)
    a.add_argument("--H", type=int, default=120)
    a.add_argument("--fill", action="store_true")
    a.add_argument("--time", type=float, default=60.0)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--missing", action="store_true")
    x = a.parse_args(argv)
    if not x.crsp:
        raise SystemExit("no CRSP extract supplied (--crsp): P6 is blocked; see docs/PLAN.md")
    code = bd.identity()
    if code["dirty"]:
        raise SystemExit("refusing to write bundles from a dirty tree; commit first")
    rows = dat.crsp(x.crsp)
    rfd = dat.rf(x.rf)
    P0, nf = bt.panel(rows, rfd, fill=x.fill)
    ip = {"crsp": rows["sha"], "rf": rfd["sha"], "rf_vintage": rfd["vintage"]}
    dates = range(x.M - 1, P0.R.shape[0] - x.H)
    for s in x.only or ST:
        key = f"oos-{s}-K{x.K}-M{x.M}-m{x.m}-Nc{x.Nc}-H{x.H}-{'fill' if x.fill else 'nofill'}"
        pa = {"s": s, "K": x.K, "M": x.M, "m": x.m, "Nc": x.Nc, "H": x.H, "fill": x.fill,
              "tl": x.time, "seed": x.seed}
        if x.missing and bd.valid(OUT, key, pa, code["source"]):
            continue
        g = np.random.default_rng(x.seed)
        o = bt.run(P0, rule(s, x.K, x.M, x.time, g), dates, x.M, x.m, x.Nc, x.H)
        r = {"mo": [q["mo"] for q in o], "W": [float(q["W"][-1]) for q in o],
             "pc": [P0.pc[q["cols"][q["w"] > 0]].tolist() for q in o],
             "B1": [float(bt.b1(P0, q["t"], x.H)[-1]) for q in o],
             "B5": [float(bt.b5(P0, q["t"], x.H)[-1]) for q in o], "fills": nf}
        bd.write(OUT, bd.bundle(key, pa, r, ip, sys.argv, code))
        print(key, len(o), "dates", flush=True)


if __name__ == "__main__":
    main()
