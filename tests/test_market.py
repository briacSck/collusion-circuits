"""Sanity tests for the Bertrand lab (design doc §5.1)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market import BertrandMarket


@pytest.fixture(scope="module")
def mkt():
    return BertrandMarket()


def test_calvano_benchmarks(mkt):
    assert mkt.nash_price == pytest.approx(1.473, abs=0.002)
    assert mkt.monopoly_price == pytest.approx(1.925, abs=0.002)


def test_nash_below_monopoly_above_cost(mkt):
    assert mkt.cost < mkt.nash_price < mkt.monopoly_price


def test_demand_sums_below_one_and_decreasing_in_own_price(mkt):
    q = mkt.demand(np.array([1.5, 1.5]))
    assert q.sum() < 1
    q_hi = mkt.demand(np.array([1.8, 1.5]))
    assert q_hi[0] < q[0]


def test_collusion_index_endpoints(mkt):
    assert mkt.collusion_index(mkt.nash_price) == pytest.approx(0.0, abs=1e-9)
    assert mkt.collusion_index(mkt.monopoly_price) == pytest.approx(1.0, abs=1e-9)


def test_static_br_is_interior_and_upward_sloping(mkt):
    br = mkt.static_br_on_grid()
    # prices are strategic complements: BR weakly increasing in rival price
    assert np.all(np.diff(br) >= 0)
    assert 0 < br[0] <= br[-1] < mkt.n_prices
