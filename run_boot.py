"""RQ1: Bessembinder's (2018) bootstrap of random K-stock portfolios on CRSP (gate G3).

    python run_boot.py --crsp PATH [--rf PATH] [--n 20000] [--sizes 1 5 25 50 100]
                       [--seed 0] [--missing]

Needs a licensed CRSP extract (pf/dat.py, COLS); without one it stops, and P5 stays blocked.
Protocol and the G3 tolerance are fixed in docs/PLAN.md: July 1926 to December 2016, share codes
10-12, CRSP delisting returns as supplied (no Shumway fill, as in the source), monthly redraw,
value weights by beginning-of-month market value. One bundle per size K in docs/results/boot/;
no row-level data is written.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from pf import bd, bs, bt, dat

# Legend
# P0 : panel; I : beginning-of-month market value; Rp : bootstrap portfolio returns
# n : replications; K : stocks per month; L : months linked for the 1- and 10-year outcomes

OUT = Path("docs/results/boot")
SPAN = (192607, 201612)
HZ = {"1y": 12, "10y": 120}


def outcomes(Rp, rf, vw, L, g):
    """Linked return per replication over L months from a start drawn uniformly per replication
    (docs/PLAN.md: the source does not state its windows), and over the whole span."""
    n, Tm = Rp.shape
    s = g.integers(0, Tm - L + 1, n)
    idx = s[:, None] + np.arange(L)
    x = np.prod(1 + np.take_along_axis(Rp, idx, axis=1), axis=1) - 1
    f = np.prod(1 + rf[idx], axis=1) - 1
    m = np.prod(1 + vw[idx], axis=1) - 1
    return x, f, m


def stats(x, f, m):
    z = (x - x.mean()) / x.std()
    return {"mean": float(x.mean()), "med": float(np.median(x)), "skew": float(np.mean(z ** 3)),
            "gt0": float(np.mean(x > 0)), "gtf": float(np.mean(x > f)),
            "gtvw": float(np.mean(x > m))}


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--crsp")
    a.add_argument("--rf")
    a.add_argument("--n", type=int, default=20000)
    a.add_argument("--sizes", type=int, nargs="+", default=[1, 5, 25, 50, 100])
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--missing", action="store_true")
    x = a.parse_args(argv)
    if not x.crsp:
        raise SystemExit("no CRSP extract supplied (--crsp): P5 is blocked; see docs/PLAN.md")
    code = bd.identity()
    if code["dirty"]:
        raise SystemExit("refusing to write bundles from a dirty tree; commit first")
    rows = dat.crsp(x.crsp)
    rfd = dat.rf(x.rf)
    P0, _ = bt.panel(rows, rfd, fill=False)
    k = (P0.mo >= SPAN[0]) & (P0.mo <= SPAN[1])
    I = np.vstack([np.full(P0.R.shape[1], np.nan), P0.ME[:-1]])[k]
    R, rf = P0.R[k], P0.rf[k]
    vw = bs.vw(R, I)
    ip = {"crsp": rows["sha"], "rf": rfd["sha"], "rf_vintage": rfd["vintage"]}
    for K in x.sizes:
        key = f"boot-K{K}"
        pa = {"K": K, "n": x.n, "seed": x.seed, "span": SPAN}
        if x.missing and bd.valid(OUT, key, pa, code["source"]):
            continue
        g = np.random.default_rng([x.seed, K])
        Rp = bs.draw(R, I, K, x.n, g)
        r = {"life": stats(np.prod(1 + Rp, axis=1) - 1,
                           np.full(x.n, np.prod(1 + rf) - 1), np.full(x.n, np.prod(1 + vw) - 1))}
        for h, L in HZ.items():
            r[h] = stats(*outcomes(Rp, rf, vw, L, g))
        bd.write(OUT, bd.bundle(key, pa, r, ip, sys.argv, code))
        print(key, r["life"], flush=True)


if __name__ == "__main__":
    main()
