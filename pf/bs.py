"""Bessembinder's (2018) bootstrap of random K-stock portfolios (sec. 4, Table 4).

For each month, K stocks are drawn at random from those with a return that month; the month's
portfolio return is value-weighted by beginning-of-month market value; months are linked. The
procedure is repeated n times (20,000 in the source) and each linked return is compared with
zero, the T-bill and the value-weighted market over the same months.

This redraws the names every month. The specification's B4 holds K names bought once, equally
weighted (pf/bt.py, b4); that is an extension, not the source's protocol. Which months the
source links for its 1- and 10-year outcomes is not stated; the windows are an argument here,
fixed in docs/PLAN.md before any run on CRSP.
"""

import numpy as np

# Legend
# R      : returns (Tm, N), nan where a stock has no return
# I      : beginning-of-month market value (Tm, N), nan where unknown
# K      : stocks drawn per month (1 in the source's single-stock case)
# n      : replications; g : numpy Generator
# Rp     : portfolio returns (n, Tm)
# wd     : windows as (first month index, last month index) inclusive
# rf, vw : T-bill and value-weighted market returns (Tm,)


def draw(R, I, K, n, g):
    """Monthly value-weighted returns of K stocks drawn afresh each month, n replications."""
    R, I = np.asarray(R, float), np.asarray(I, float)
    Tm = R.shape[0]
    Rp = np.empty((n, Tm))
    for t in range(Tm):
        a = np.flatnonzero(np.isfinite(R[t]) & np.isfinite(I[t]) & (I[t] > 0))
        if a.size < K:
            raise ValueError(f"month {t}: {a.size} stocks available, {K} drawn")
        c = np.array([g.choice(a, size=K, replace=False) for _ in range(n)])
        v = I[t][c]
        Rp[:, t] = np.sum(v * R[t][c], axis=1) / v.sum(axis=1)
    return Rp


def vw(R, I):
    """Value-weighted market return each month over stocks with a return and a value."""
    R, I = np.asarray(R, float), np.asarray(I, float)
    k = np.isfinite(R) & np.isfinite(I) & (I > 0)
    return np.where(k, I * np.nan_to_num(R), 0).sum(axis=1) / np.where(k, I, 0).sum(axis=1)


def link(Rp, wd):
    """Linked returns prod(1 + r) - 1 over each window, per replication: (n, len(wd))."""
    Rp = np.atleast_2d(Rp)
    return np.stack([np.prod(1 + Rp[:, a:b + 1], axis=1) - 1 for a, b in wd], axis=1)


def table(Rp, rf, vwm, wd):
    """Table 4 rows for the windows wd: mean, median, skewness, and shares > 0, > T-bill, > VW."""
    x = link(Rp, wd).ravel()
    f = np.repeat(link(rf, wd), Rp.shape[0], axis=0).ravel()
    m = np.repeat(link(vwm, wd), Rp.shape[0], axis=0).ravel()
    z = (x - x.mean()) / x.std()
    return {"mean": float(x.mean()), "med": float(np.median(x)), "skew": float(np.mean(z ** 3)),
            "gt0": float(np.mean(x > 0)), "gtf": float(np.mean(x > f)),
            "gtvw": float(np.mean(x > m)), "n": int(x.size)}
