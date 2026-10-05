"""Regression fixtures for point-in-time composition and financial arithmetic."""

import numpy as np
import pytest

from pf import bs, bt, dat, ev, wc
from run_oos import main as oos_main
from run_oos import rule, window


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


def test_delisted_name_cannot_be_bought_again_at_selection():
    p = bt.P([200001, 200002, 200003], [10], [1],
             [[.1], [-.5], [np.nan]], [[10], [5], [np.nan]], [.01] * 3, ls=[1])
    assert bt.elig(p.upto(1), 2, 2, 1).size == 0
    q = bt.run(p, bt.b2, [1], 2, 2, 1, 1)[0]
    assert q["status"] == "unavailable" and q["cols"].size == 0
    assert q["W"].size == 0
    np.testing.assert_allclose(bt.hold(p, 0, [0], [1], 2), [.5, .505])


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


@pytest.mark.parametrize("scale", [1.01, 10.0])
def test_positive_rescaling_has_zero_sharpe_difference(scale):
    r = np.array([.1, .2, -.1, .05, .15, -.15, .07, .01])
    x = ev.lw(r, scale * r)
    assert x == {"D": 0.0, "se": 0.0, "p": 1.0, "ci": [0.0, 0.0]}


def test_small_actual_sharpe_difference_survives_roundoff_check():
    r = np.array([.1, .2, -.1, .05, .15, -.15, .07, .01])
    altered = 1.01 * r
    altered[0] += 1e-9
    x = ev.lw(r, altered)
    assert x["D"] != 0 and x["se"] > 0


@pytest.mark.parametrize("field,value", [("DATE", "20001332"), ("PERMNO", "1.5"),
                                          ("RET", "inf"), ("RET", "-1.01")])
def test_crsp_rejects_invalid_cells_without_echoing_row(tmp_path, field, value):
    row = dict(zip(dat.COLS, ["1", "1", "20000131", "10", "1", "10", "100", ".1", "", ""]))
    row[field] = value
    p = tmp_path / "hand.csv"
    p.write_text(",".join(dat.COLS) + "\n" + ",".join(row[c] for c in dat.COLS) + "\n")
    with pytest.raises(dat.DataError) as e:
        dat.crsp(p)
    assert value not in str(e.value)


def test_cutoff_columns_map_back_to_the_holding_panel():
    p = bt.P([200001, 200002, 200003], [10, 20], [1, 2],
             [[np.nan, .1], [np.nan, .2], [.9, .3]],
             [[np.nan, 10], [np.nan, 11], [20, 12]], [0, 0, 0])
    q = bt.run(p, lambda pt, u: (u, [1]), [1], 2, 2, 2, 1)[0]
    assert q["cols"].tolist() == [1]
    assert q["pc"].tolist() == [2]
    np.testing.assert_allclose(q["W"], [1.3])


def test_missing_selection_dates_remain_in_the_sample():
    p = bt.P([200001, 200002, 200003], [10], [1],
             [[np.nan], [.1], [.2]], [[np.nan], [10], [11]], [0, 0, 0])
    out = bt.run(p, lambda pt, u: (u, [1]), [0, 1], 1, 1, 1, 1)
    assert [q["mo"] for q in out] == [200001, 200002]
    assert out[0]["status"] == "unavailable" and out[0]["W"].size == 0
    assert out[1]["status"] == "ok"


@pytest.mark.parametrize("s", ["f2", "f3"])
def test_risk_strategies_use_registered_return_floor_and_keep_certificate(monkeypatch, s):
    p = bt.P([200001, 200002], [1, 2], [1, 2],
             [[.1, .2], [.3, .4]], np.ones((2, 2)), [0, 0])
    seen = {}

    def solve(y, K, **kwargs):
        seen.update(kwargs)
        return {"w": np.array([.5, .5]), "c": {"st": "limit", "g": .1}}

    module = "run_oos.mad.mad" if s == "f2" else "run_oos.cv.cv"
    monkeypatch.setattr(module, solve)
    q = rule(s, 2, 2, 1, np.random.default_rng(0))(p, np.array([0, 1]))
    assert seen["rho"] == pytest.approx(.25)
    assert q["status"] == "limit" and q["c"]["g"] == .1


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        rule("typo", 1, 2, 1, np.random.default_rng(0))


def test_oos_runner_keeps_failed_dates_and_hashes_inputs(monkeypatch):
    p = bt.P([200001, 200002, 200003], [10], [1],
             [[np.nan], [.1], [.2]], [[np.nan], [10], [11]], [0, 0, 0])
    saved, reused = [], []
    monkeypatch.setattr("run_oos.bd.identity", lambda: {"dirty": False, "source": "hand"})
    monkeypatch.setattr("run_oos.dat.crsp", lambda path: {"sha": "crsp-hand"})
    monkeypatch.setattr("run_oos.dat.rf", lambda path: {"sha": "rf-hand", "vintage": "hand"})
    monkeypatch.setattr("run_oos.bt.panel", lambda rows, rf, fill: (p, 0))
    monkeypatch.setattr("run_oos.bd.valid", lambda *args: reused.append(args))
    monkeypatch.setattr("run_oos.bd.bundle", lambda key, pa, r, ip, cmd, code: (r, ip, cmd))
    monkeypatch.setattr("run_oos.bd.write", lambda out, bundle: saved.append(bundle))
    argv = ["--crsp", "hand", "--only", "b2", "--M", "1", "--m", "1", "--H", "1",
            "--K", "1", "--Nc", "1", "--missing"]
    oos_main(argv)
    r, ip, cmd = saved[0]
    assert r["mo"] == [200001, 200002]
    assert r["W"][0] is None and r["W"][1] == pytest.approx(1.2)
    assert r["selection"][0]["status"] == "unavailable"
    assert r["B1"][0] is None and "market" in r["B1_errors"][0]
    assert r["B1"][1] == pytest.approx(1.2) and r["B1_errors"][1] is None
    assert reused[0][-1] == ip and len(ip["plan"]) == 64
    assert cmd == ["run_oos.py", *argv]


def test_f5_rule_uses_registered_utility_and_shrinkage(monkeypatch):
    p = bt.P([200001, 200002, 200003, 200004], [1, 2], [1, 2],
             [[.1, .2], [.3, .1], [-.2, .1], [.4, .3]], np.ones((4, 2)), np.zeros(4))
    seen = {}

    def solve(y, K, g, **kwargs):
        seen.update(y=y, K=K, g=g, **kwargs)
        return {"w": np.array([.5, .5]), "c": {"st": "limit", "g": .2}}

    monkeypatch.setattr("run_oos.sk.f5", solve)
    q = rule("f5", 2, 4, 1, np.random.default_rng(0))(p, np.array([0, 1]))
    assert seen["g"] == 3 and seen["k"] == 12 and seen["Sg"].shape == (2, 2)
    np.testing.assert_array_equal(seen["y"], p.R)
    assert q["c"]["st"] == "limit" and q["rho"] is None


@pytest.mark.parametrize("s", ["b3", "b4", "f1", "f2", "f3", "f5"])
def test_insufficient_universe_is_recorded_for_each_k_strategy(s):
    p = bt.P([200001, 200002, 200003], [1], [1],
             [[.1], [.2], [.3]], np.ones((3, 1)), [0, 0, 0])
    q = bt.run(p, rule(s, 2, 2, 1, np.random.default_rng(0)), [1], 2, 2, 2, 1)[0]
    assert q["status"] == "unavailable" and "fewer than K" in q["reason"]
