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
# prev, first : preceding valuation month and initial beginning-of-month values

OUT = Path("docs/results/boot")
SPAN = (192607, 201612)
HZ = {"1y": 12, "10y": 120}


def outcomes(Rp, rf, vw, L, g):
    """Linked return per replication over L months from a start drawn uniformly per replication
    (docs/PLAN.md: the source does not state its windows), and over the whole span."""
    n, Tm = Rp.shape
    if (L < 1 or L > Tm or np.asarray(rf).shape != (Tm,)
            or np.asarray(vw).shape != (Tm,)):
        raise ValueError("bootstrap horizon or benchmark shape is invalid")
    s = g.integers(0, Tm - L + 1, n)
    idx = s[:, None] + np.arange(L)
    x = np.prod(1 + np.take_along_axis(Rp, idx, axis=1), axis=1) - 1
    f = np.prod(1 + rf[idx], axis=1) - 1
    m = np.prod(1 + vw[idx], axis=1) - 1
    return x, f, m


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
    if x.n < 1 or min(x.sizes) < 1 or x.seed < 0:
        a.error("positive draw sizes and replications, and a nonnegative seed, are required")
    code = bd.identity()
    if code["dirty"]:
        raise SystemExit("refusing to write bundles from a dirty tree; commit first")
    rows = dat.crsp(x.crsp)
    rfd = dat.rf(x.rf)
    y, m = divmod(SPAN[0], 100)
    prev = (y - 1) * 100 + 12 if m == 1 else y * 100 + m - 1
    p = rows["mo"] == prev
    if not p.any():
        raise SystemExit(f"CRSP extract needs the preceding month's valuation rows ({prev})")
    k = (rows["mo"] >= SPAN[0]) & (rows["mo"] <= SPAN[1])
    if not np.array_equal(np.unique(rows["mo"][k]), dat.mseq(*SPAN)):
        raise SystemExit("CRSP extract must cover the complete registered replication span")
    cut = {c: v[k] if isinstance(v, np.ndarray) else v for c, v in rows.items()}
    P0, _ = bt.panel(cut, rfd, fill=False)
    first = dict(zip(rows["PERMNO"][p], np.abs(rows["PRC"][p]) * rows["SHROUT"][p]))
    initial = np.array([first.get(i, np.nan) for i in P0.ids])
    if not np.any(np.isfinite(initial) & (initial > 0) & np.isfinite(P0.R[0])):
        raise SystemExit("CRSP extract needs usable preceding-month valuations")
    I = np.vstack([initial, P0.ME[:-1]])
    R, rf = P0.R, P0.rf
    vw = bs.vw(R, I)
    ip = {"crsp": rows["sha"], "rf": rfd["sha"], "rf_vintage": rfd["vintage"],
          "plan": dat.sha256(Path(__file__).parent / "docs/PLAN.md")}
    for K in x.sizes:
        key = f"boot-K{K}"
        pa = {"K": K, "n": x.n, "seed": x.seed, "span": SPAN, "valuation": prev}
        if x.missing and bd.valid(OUT, key, pa, code["source"], ip):
            continue
        g = np.random.default_rng([x.seed, K])
        Rp = bs.draw(R, I, K, x.n, g)
        r = {"life": bs.stats(np.prod(1 + Rp, axis=1) - 1,
                              np.full(x.n, np.prod(1 + rf) - 1), np.full(x.n, np.prod(1 + vw) - 1))}
        for h, L in HZ.items():
            r[h] = bs.stats(*outcomes(Rp, rf, vw, L, g))
        cmd = sys.argv if argv is None else [str(Path(__file__).name), *argv]
        bd.write(OUT, bd.bundle(key, pa, r, ip, cmd, code))
        print(key, r["life"], flush=True)


if __name__ == "__main__":
    main()
