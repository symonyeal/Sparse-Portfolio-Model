"""Delisting arithmetic, wealth creation, the engine and inference, on hand fixtures."""

import ast
from pathlib import Path

import numpy as np
import pytest

from pf import bs, bt, ev, mv, wc

# Legend
# P0 : a hand panel; t : rebalance index; sel : a selection rule over P.upto(t)

ROOT = Path(__file__).resolve().parent.parent


def test_delisting_combination_and_shumway_fill():
    ret = np.array([0.10, np.nan, 0.05, np.nan, np.nan])
    dl = np.array([-0.50, -0.20, np.nan, np.nan, np.nan])
    dc = np.array([551, 233, 0, 560, 560])
    ex = np.array([1, 1, 1, 1, 3])
    R, n = wc.comb(ret, dl, dc, ex)
    assert np.allclose(R[:3], [1.1 * 0.5 - 1, -0.20, 0.05]) and np.isnan(R[3:]).all() and n == 0
    R, n = wc.comb(ret, dl, dc, ex, fill=True)
    assert np.allclose(R[3:], [-0.30, -0.55]) and n == 2
    assert wc.perf([500, 519, 520, 584, 585], [1] * 5).tolist() == [1, 0, 1, 1, 0]
    assert wc.perf([505, 588, 589], [3] * 3).tolist() == [1, 1, 0]


def test_wealth_creation_identity_against_accounts():
    """Bessembinder (2018) eq. (3) against his account recursion, with dividends and issues."""
    rf = np.array([0.01, 0.02, 0.005, 0.01])
    Rc = np.array([0.10, -0.30, 0.20, 0.05])          # capital gains
    Rd = np.array([0.02, 0.01, 0.00, 0.03])           # dividends
    F = np.array([0.0, 5.0, -3.0, 0.0])               # issues (+) and repurchases (-)
    B, I, W0 = 50.0, 100.0, 150.0
    Ib = []
    for t in range(4):
        Ib.append(I)
        B = B * (1 + rf[t]) + I * Rd[t] - F[t]
        I = I * (1 + Rc[t]) + F[t]
    lhs = (B + I) - W0 * np.prod(1 + rf)
    assert np.isclose(wc.wc(np.array(Ib), Rc + Rd, rf), lhs, rtol=1e-13)
    with pytest.raises(ValueError):
        wc.wc(np.array([1.0, np.nan]), np.array([0.1, 0.1]), np.array([0.0, 0.0]))


def test_top_set_and_firm_sums():
    u, W = wc.firm([5, -2, 3, 1, -1], [7, 7, 8, 9, 10])
    assert u.tolist() == [7, 8, 9, 10] and W.tolist() == [3, 3, 1, -1]
    assert sorted(wc.top(W).tolist()) == [0, 1]          # 3 + 3 >= net 6
    assert wc.top(np.array([4.0, -1.0, 2.0])).tolist() == [0, 2]   # 4 < net 5 <= 4 + 2
    assert wc.top(np.array([6.0, -1.0, 0.5])).tolist() == [0]      # 6 >= net 5.5


def hand(n=12, N=4, seed=0):
    g = np.random.default_rng(seed)
    R = g.normal(0.01, 0.05, (n, N))
    R[8:, 3] = np.nan                                  # stock 3 delists after month 7
    ME = np.where(np.isnan(R), np.nan, g.uniform(1, 10, (n, N)))
    return bt.P(np.arange(n) + 200001, np.arange(N) + 1, np.arange(N) + 1, R, ME,
                np.full(n, 0.002))


def test_hold_moves_delisted_proceeds_to_cash():
    P0 = hand()
    W = bt.hold(P0, 5, [0, 3], [0.5, 0.5], 4)
    a = 0.5 * np.prod(1 + P0.R[6:10, 0])
    b = 0.5 * np.prod(1 + P0.R[6:8, 3]) * 1.002 ** 2
    assert np.isclose(W[-1], a + b)


def test_missing_return_while_listed_is_refused():
    P0 = hand()
    P0.R[6, 1] = np.nan
    with pytest.raises(ValueError, match="still listed"):
        bt.hold(P0, 5, [1], [1.0], 3)


def test_no_look_ahead():
    """Perturbing every return after t leaves the selection at t unchanged."""
    def sel(Pt, U):
        y = Pt.R[-4:][:, U]
        S, _ = mv.shrink(y)
        k = U[np.argsort(np.diag(S))[:2]]
        return k, np.full(2, 0.5)
    P0, P1 = hand(), hand()
    t = 6
    P1.R[t + 1:] += np.random.default_rng(9).normal(0, 0.1, P1.R[t + 1:].shape)
    P1.ME[t + 1:] *= 3
    a = bt.run(P0, sel, [t], 4, 4, 3, 2)[0]
    b = bt.run(P1, sel, [t], 4, 4, 3, 2)[0]
    assert np.array_equal(a["cols"], b["cols"]) and np.array_equal(a["w"], b["w"])
    assert not np.allclose(a["W"], b["W"])


def test_future_labels_never_reach_selection():
    """No selection or engine module refers to the wealth-creation set or the evaluation module."""
    for m in ("bt", "rw", "mv", "ecp", "mad", "cv", "it", "sk"):
        tree = ast.parse((ROOT / "pf" / f"{m}.py").read_text(encoding="utf-8"))
        imp = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        use = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        use |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert "ev" not in imp and "top" not in use and "firm" not in use, m


def test_bessembinder_draw_of_all_stocks_is_the_market():
    P0 = hand()
    I = np.vstack([np.full(4, np.nan), P0.ME[:-1]])
    R, I = P0.R[1:], I[1:]
    k = np.isfinite(R) & np.isfinite(I)
    R2 = np.where(k, R, np.nan)
    n = int(k.sum(axis=1).min())
    one = bs.draw(R2[:, :3], I[:, :3], 3, 2, np.random.default_rng(0))
    assert np.allclose(one[0], bs.vw(R2[:, :3], I[:, :3])) and n >= 3
    assert np.allclose(bs.link(one, [(0, 1)])[:, 0], (1 + one[:, 0]) * (1 + one[:, 1]) - 1)


def test_stationary_bootstrap_and_sharpe_test():
    g = np.random.default_rng(1)
    i = ev.sbi(1000, 0.1, g)
    assert i.min() >= 0 and i.max() < 1000
    runs = np.mean(np.diff(i) == 1)
    assert 0.85 < runs < 0.95                          # continuation probability 1 - p
    x, y = g.normal(0.01, 0.04, 240), g.normal(0.01, 0.04, 240)
    o = ev.lw(x, y)
    sx = x.mean() / x.std()
    sy = y.mean() / y.std()
    assert np.isclose(o["D"], sx - sy) and 0 <= o["p"] <= 1 and o["se"] > 0
    v = np.array([x.mean(), y.mean(), np.mean(x ** 2), np.mean(y ** 2)])
    h = 1e-7
    fd = [(ev._f(v + h * e) - ev._f(v - h * e)) / (2 * h) for e in np.eye(4)]
    assert np.allclose(fd, ev._df(v), rtol=1e-5)


def test_underperformance_and_hit_rate():
    assert ev.under([1.0, 2.0, 0.5], [1.5, 1.5, 1.5]) == pytest.approx(2 / 3)
    assert ev.hit([[1, 2], [3], [4, 5]], [2, 5]) == pytest.approx(2 / 3)
    assert ev.cvw(np.array([1.2, 0.8, 0.5, 1.0]), 0.5) == pytest.approx(0.35)
