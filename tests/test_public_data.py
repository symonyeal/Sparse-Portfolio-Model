"""Malformed public data are rejected before optimization; OR-Library and French formats.

Legend: p is a fixture path; s is its text; k identifies an OR-Library instance.
"""

import pytest

from pf import dat


@pytest.mark.parametrize("s", ["1.5 1\n100 101 100 101", "1 1.5\n100 101 100 101"])
def test_integer_headers(pth, s):
    pth.write_text(s)
    with pytest.raises(dat.DataError, match="integer"):
        dat.indtrack(1, pth)


@pytest.fixture
def pth(tmp_path):
    return tmp_path / "fixture.txt"


def test_only_documented_extra_series(pth):
    pth.write_text("1 1\n100 101 100 101 rubbish")
    with pytest.raises(dat.DataError):
        dat.indtrack(1, pth, tail=True)


def test_empty_frontier(pth):
    pth.write_text("")
    with pytest.raises(dat.DataError, match="empty"):
        dat.portef(1, pth)


@pytest.mark.parametrize("s", [",Mkt-RF,SMB,HML,RF\n\n",
                              ",Mkt-RF,SMB,HML,RF\n202613,1,1,1,.1\n\n"])
def test_empty_or_invalid_rf(pth, s):
    pth.write_text(s)
    with pytest.raises(dat.DataError):
        dat.rf(pth)
