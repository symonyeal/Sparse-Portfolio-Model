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
# Ps     : HAC estimate of Psi; bw : kernel bandwidth S_T; D : Sharpe difference


def under(Ws, Wb):
    """Share of start dates with strategy wealth below the benchmark's."""
    return float(np.mean(np.asarray(Ws) < np.asarray(Wb)))


def hit(hs, T):
    """Share of portfolios holding at least one firm of T."""
    T = set(np.asarray(T).tolist())
    return float(np.mean([bool(T.intersection(np.asarray(h).tolist())) for h in hs]))


def skew(x):
    """Standardised third moment."""
    x = np.asarray(x, float)
    return float(np.mean(((x - x.mean()) / x.std()) ** 3))


def sharpe(r):
    """Mean over standard deviation of excess returns."""
    r = np.asarray(r, float)
    return float(r.mean() / r.std(ddof=1))


def cvw(W, b=0.95):
    """beta-CVaR of the terminal-wealth loss 1 - W, equal weights."""
    W = np.asarray(W, float)
    return cv.cvar((W - 1.0)[:, None], np.ones(1), b)


def sbi(n, p, g):
    """Indices of one stationary-bootstrap resample of 0..n-1 (Politis and Romano, 1994)."""
    i = np.empty(n, dtype=int)
    i[0] = g.integers(n)
    for t in range(1, n):
        i[t] = g.integers(n) if g.random() < p else (i[t - 1] + 1) % n
    return i


def sbci(x, p, B, g, al=0.05):
    """Mean of x and its stationary-bootstrap percentile interval."""
    x = np.asarray(x, float)
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
    from math import erf, sqrt
    ri, rn = np.asarray(ri, float), np.asarray(rn, float)
    T = ri.size
    v = np.array([ri.mean(), rn.mean(), np.mean(ri ** 2), np.mean(rn ** 2)])
    y = np.c_[ri - v[0], rn - v[1], ri ** 2 - v[2], rn ** 2 - v[3]]
    bw = 4 * (T / 100) ** (2 / 9) if bw is None else bw
    Ps = np.zeros((4, 4))
    for j in range(-(T - 1), T):
        k = parzen(j / bw)
        if k == 0:
            continue
        G = y[j:].T @ y[:T - j] / T if j >= 0 else y[:T + j].T @ y[-j:] / T
        Ps += k * G
    Ps *= T / (T - 4)
    gr = _df(v)
    se = float(np.sqrt(gr @ Ps @ gr / T))
    D = float(_f(v))
    z = abs(D) / se
    return {"D": D, "se": se, "p": float(2 * (1 - 0.5 * (1 + erf(z / sqrt(2)))))}
