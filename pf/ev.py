"""Evaluation: underperformance probabilities, hit rate, terminal-wealth metrics, inference.

Primary metric (specification sec. 8): the share of start dates at which a strategy's
buy-and-hold wealth over H falls below B1's, and below B5's. Inference: the stationary bootstrap
(Politis and Romano, 1994) for differences over overlapping windows, and the HAC test of
Ledoit and Wolf (2008, sec. 3.1) for a difference of Sharpe ratios.

The set T of top wealth creators uses data after every selection date; it enters here and
nowhere in selection (a test asserts that pf/bt.py and the selection modules never import it).
"""

import numpy as np

from pf import cv

# Legend
# Ws, Wb : terminal wealth of a strategy and of a benchmark per start date
# hs     : held PERMCOs per start date; T : PERMCOs of the top wealth creators
# p      : stationary-bootstrap restart probability (mean block length 1/p)
# B      : bootstrap replications; g : numpy Generator; al : two-sided level
# ri, rn : excess returns of two strategies over the same months
# v      : (mu_i, mu_n, gamma_i, gamma_n), first and uncentred second moments
# u      : scalar influence y @ gradient; bw : kernel bandwidth S_T; D : Sharpe difference
# su, sd : roundoff scales of the influence and Sharpe difference
# hac    : HAC variance; a, nq : absolute HAC sum and number of lag terms


def vector(x):
    """A nonempty finite observation vector."""
    x = np.asarray(x, float)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("observations must be a nonempty finite vector")
    return x


def under(Ws, Wb):
    """Share of start dates with strategy wealth below the benchmark's."""
    Ws, Wb = vector(Ws), vector(Wb)
    if Ws.shape != Wb.shape:
        raise ValueError("wealth observations must be paired")
    return float(np.mean(Ws < Wb))


def hit(hs, T):
    """Share of portfolios holding at least one firm of T."""
    T = set(np.asarray(T).tolist())
    if len(hs) == 0:
        raise ValueError("hit rate needs at least one portfolio")
    return float(np.mean([bool(T.intersection(np.asarray(h).tolist())) for h in hs]))


def skew(x):
    """Standardised third moment."""
    x = vector(x)
    if x.std() == 0:
        raise ValueError("skewness is undefined for constant observations")
    return float(np.mean(((x - x.mean()) / x.std()) ** 3))


def sharpe(r):
    """Mean over standard deviation of excess returns."""
    r = vector(r)
    if r.size < 2 or r.std(ddof=1) == 0:
        raise ValueError("Sharpe ratio needs at least two nonconstant returns")
    return float(r.mean() / r.std(ddof=1))


def cvw(W, b=0.95):
    """beta-CVaR of the terminal-wealth loss 1 - W, equal weights."""
    W = vector(W)
    if not 0 < b < 1:
        raise ValueError("CVaR beta must be between zero and one")
    return cv.cvar((W - 1.0)[:, None], np.ones(1), b)


def sbi(n, p, g):
    """Indices of one stationary-bootstrap resample of 0..n-1 (Politis and Romano, 1994)."""
    if not isinstance(n, (int, np.integer)) or n < 1 or not 0 < p <= 1:
        raise ValueError("bootstrap needs a positive size and restart probability in (0, 1]")
    i = np.empty(n, dtype=int)
    i[0] = g.integers(n)
    for t in range(1, n):
        i[t] = g.integers(n) if g.random() < p else (i[t - 1] + 1) % n
    return i


def sbci(x, p, B, g, al=0.05):
    """Mean of x and its stationary-bootstrap percentile interval."""
    x = vector(x)
    if not isinstance(B, (int, np.integer)) or B < 1 or not 0 < al < 1:
        raise ValueError("bootstrap replications and interval level are invalid")
    m = np.array([x[sbi(x.size, p, g)].mean() for _ in range(B)])
    return float(x.mean()), (float(np.quantile(m, al / 2)), float(np.quantile(m, 1 - al / 2)))


def _f(v):
    a, b, c, d = v
    return a / np.sqrt(c - a * a) - b / np.sqrt(d - b * b)


def _df(v):
    a, b, c, d = v
    return np.array([c / (c - a * a) ** 1.5, -d / (d - b * b) ** 1.5,
                     -0.5 * a / (c - a * a) ** 1.5, 0.5 * b / (d - b * b) ** 1.5])


def parzen(x):
    """Parzen kernel."""
    x = abs(x)
    return 1 - 6 * x * x + 6 * x ** 3 if x <= 0.5 else (2 * (1 - x) ** 3 if x <= 1 else 0.0)


def lw(ri, rn, bw=None):
    """Sharpe difference, its HAC standard error and two-sided p-value (Ledoit and Wolf, 2008,
    eqs. (1)-(5), Parzen kernel). bw defaults to 4 (T/100)^(2/9), a project setting; the paper
    leaves the bandwidth to Andrews (1991) and prefers its studentized bootstrap."""
    from math import erfc, sqrt
    ri, rn = vector(ri), vector(rn)
    if ri.shape != rn.shape or ri.size <= 4 or ri.std() == 0 or rn.std() == 0:
        raise ValueError("Sharpe inference needs paired nonconstant returns and more than 4 months")
    T = ri.size
    v = np.array([ri.mean(), rn.mean(), np.mean(ri ** 2), np.mean(rn ** 2)])
    y = np.c_[ri - v[0], rn - v[1], ri ** 2 - v[2], rn ** 2 - v[3]]
    bw = 4 * (T / 100) ** (2 / 9) if bw is None else bw
    if not np.isfinite(bw) or bw <= 0:
        raise ValueError("HAC bandwidth must be finite and positive")
    gr = _df(v)
    u = y @ gr
    D = float(_f(v))
    eps = np.finfo(float).eps
    su = np.abs(y) @ np.abs(gr)
    sd = abs(v[0] / ri.std()) + abs(v[1] / rn.std())
    # Both the difference and its influence must vanish at floating-point precision.
    if np.all(np.abs(u) <= 16 * eps * su) and abs(D) <= 16 * eps * sd:
        return {"D": 0.0, "se": 0.0, "p": 1.0, "ci": [0.0, 0.0]}
    hac, a, nq = 0.0, 0.0, 0
    for j in range(T):
        k = parzen(j / bw)
        if k == 0:
            continue
        k *= 1 if j == 0 else 2
        hac += k * float(u[j:] @ u[:T - j]) / T
        a += k * float(np.abs(u[j:]) @ np.abs(u[:T - j])) / T
        nq += 1
    if hac < -eps * (T + nq + 1) * a:
        raise ValueError("HAC variance is negative")
    hac = max(hac, 0.0) * T / (T - 4)
    var = hac / T
    se = float(np.sqrt(var))
    z = abs(D) / se if se else (0.0 if D == 0 else np.inf)
    return {"D": D, "se": se, "p": float(erfc(z / sqrt(2))),
            "ci": [D - 1.959963984540054 * se, D + 1.959963984540054 * se]}
