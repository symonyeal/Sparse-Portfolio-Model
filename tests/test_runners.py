"""Replication uses actual preceding valuations without requiring preceding T-bill returns."""

import numpy as np
import pytest

import run_boot
from pf import bd, dat

# p, out : synthetic input and bundle paths; prior : preceding valuation row


@pytest.mark.parametrize("prior", [False, True])
def test_boot_prior(tmp_path, monkeypatch, prior):
    p = tmp_path / "hand.csv"
    rows = ["1,1,20000131,10,1,11,100,.1,,", "1,1,20000229,10,1,13.2,100,.2,,"]
    if prior:
        rows.insert(0, "1,1,19991231,10,1,10,100,.05,,")
    p.write_text(",".join(dat.COLS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    rf = tmp_path / "rf.csv"
    rf.write_text(",Mkt-RF,SMB,HML,RF\n200001,0,0,0,0\n200002,0,0,0,0\n\n",
                  encoding="utf-8")
    out = tmp_path / "bundles"
    monkeypatch.setattr(run_boot, "SPAN", (200001, 200002))
    monkeypatch.setattr(run_boot, "HZ", {"1m": 1})
    monkeypatch.setattr(run_boot, "OUT", out)
    monkeypatch.setattr(bd, "identity", lambda: {"dirty": False, "source": "hand",
                                                "commit": "0" * 40})
    args = ["--crsp", str(p), "--rf", str(rf), "--sizes", "1", "--n", "2"]
    if not prior:
        with pytest.raises(SystemExit, match="preceding.*valuation"):
            run_boot.main(args)
    else:
        run_boot.main(args)
        b = bd.read(out / "boot-K1.json")
        assert b["life"]["mean"] == pytest.approx(.32)
        assert b["bundle"]["inputs"]["crsp"] == dat.sha256(p)
        assert np.isfinite(b["life"]["mean"])
