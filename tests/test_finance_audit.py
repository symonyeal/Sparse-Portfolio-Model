"""Regression fixtures for point-in-time composition and financial arithmetic."""

import numpy as np
import pytest

from pf import bs, bt, dat, ev, wc
from run_oos import window


def test_cutoff_hides_future_entrant_and_future_company():
    p = bt.P([200001, 200002, 200003], [10, 20],
             [[1, 0], [1, 0], [9, 2]],
             [[.1, np.nan], [.2, np.nan], [.3, .4]],
             [[10, np.nan], [11, np.nan], [12, 20]], [0, 0, 0])
    q = p.upto(1)
    assert q.ids.tolist() == [10]
    assert q.pc.tolist() == [1]


def test_unknown_exit_is_not_a_delisting():
    p = bt.P([200001, 200002, 200003], [10], [1],
             [[.1], [np.nan], [np.nan]], [[10], [np.nan], [np.nan]], [0, 0, 0])
    with pytest.raises(ValueError, match="missing return"):
        bt.hold(p, 0, [0], [1], 2)


def test_missing_calendar_month_is_rejected():
    with pytest.raises(ValueError, match="consecutive"):
        bt.P([200001, 200003], [1], [1], [[0], [0]], [[1], [1]], [0, 0])


def test_history_threshold_uses_common_observed_scenarios():
    p = bt.P([200001, 200002, 200003, 200004], [1, 2], [1, 2],
             [[.1, np.nan], [.2, np.nan], [.3, .4], [.4, .5]],
             np.ones((4, 2)), np.zeros(4))
    c, y = window(p, np.array([0, 1]), 4, m=2)
    assert c.tolist() == [0, 1]
    np.testing.assert_allclose(y, [[.3, .4], [.4, .5]])


def test_incomplete_holding_horizon_is_not_silently_shortened():
    p = bt.P([200001, 200002], [1], [1], [[0], [.1]], [[1], [1]], [0, 0])
    with pytest.raises(ValueError, match="horizon"):
        bt.hold(p, 0, [0], [1], 2)


def test_top_set_for_nonpositive_net_wealth_is_empty():
    assert wc.top([0, -1]).size == 0
    assert wc.top([]).size == 0


def test_bad_return_and_bootstrap_shape_are_rejected():
    with pytest.raises(ValueError, match="return"):
        wc.comb([-.2], [-1.2], [500], [1])
    with pytest.raises(ValueError):
        bs.draw(np.zeros((2, 2)), np.ones((2, 1)), 1, 3, np.random.default_rng(0))


def test_unavailable_market_month_is_rejected():
    with pytest.raises(ValueError, match="market"):
        bs.vw([[np.nan]], [[1]])


def test_inference_rejects_invalid_pair_and_constant_returns():
    with pytest.raises(ValueError):
        ev.under([1, np.nan], [1, 2])
    with pytest.raises(ValueError):
        ev.lw(np.ones(10), np.arange(10))
    with pytest.raises(ValueError):
        ev.sbi(10, 0, np.random.default_rng(0))


def test_same_strategy_sharpe_difference_has_zero_interval():
    r = np.array([.1, .2, -.1, .05, .15, -.15, .07, .01])
    x = ev.lw(r, r)
    assert x["D"] == 0 and x["p"] == 1
    assert x["ci"] == [0, 0]


@pytest.mark.parametrize("field,value", [("DATE", "20001332"), ("PERMNO", "1.5"),
                                          ("RET", "inf"), ("RET", "-1.01")])
def test_crsp_rejects_invalid_cells_without_echoing_row(tmp_path, field, value):
    row = dict(zip(dat.COLS, ["1", "1", "20000131", "10", "1", "10", "100", ".1", "", ""]))
    row[field] = value
    p = tmp_path / "hand.csv"
    p.write_text(",".join(dat.COLS) + "\n" + ",".join(row[c] for c in dat.COLS) + "\n")
    with pytest.raises(dat.DataError):
        dat.crsp(p)
