"""Repeated Bertrand duopoly with logit demand (Calvano et al. 2020 setup).

Demand for firm i at prices (p_1, ..., p_n), with an outside option:

    q_i = exp((a - p_i)/mu) / [ sum_j exp((a - p_j)/mu) + exp(a0/mu) ]

Profits pi_i = (p_i - c) * q_i. Benchmarks computed numerically:
- p_N: symmetric one-shot Bertrand-Nash price (best-response iteration);
- p_M: symmetric joint-profit-maximizing (monopoly) price.
The collusion index of an outcome with average price p is
Delta = (p - p_N) / (p_M - p_N): 0 = competitive, 1 = full collusion.

Price grid: m points spanning [p_N - xi*(p_M - p_N), p_M + xi*(p_M - p_N)]
(Calvano et al. use xi = 0.1, m = 15).
"""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class BertrandMarket:
    a: float = 2.0      # product quality
    a0: float = 0.0     # outside-option quality
    mu: float = 0.25    # horizontal differentiation
    cost: float = 1.0   # marginal cost
    n_firms: int = 2
    n_prices: int = 15
    xi: float = 0.1

    def demand(self, prices: np.ndarray) -> np.ndarray:
        """Vector of demands q_i at the price profile `prices`."""
        u = np.exp((self.a - np.asarray(prices, dtype=float)) / self.mu)
        denom = u.sum() + np.exp(self.a0 / self.mu)
        return u / denom

    def profits(self, prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=float)
        return (prices - self.cost) * self.demand(prices)

    def best_response(self, rival_price: float) -> float:
        """Continuous best response to a symmetric rival price."""
        res = minimize_scalar(
            lambda p: -self.profits(np.array([p, rival_price]))[0],
            bounds=(self.cost, self.a + 2), method="bounded",
            options={"xatol": 1e-10})
        return float(res.x)

    @property
    def nash_price(self) -> float:
        """Symmetric one-shot Nash by best-response iteration."""
        p = self.cost + 0.5
        for _ in range(200):
            p_new = self.best_response(p)
            if abs(p_new - p) < 1e-9:
                return p_new
            p = p_new
        raise RuntimeError("Nash iteration did not converge")

    @property
    def monopoly_price(self) -> float:
        """Symmetric joint-profit-maximizing price."""
        res = minimize_scalar(
            lambda p: -self.profits(np.array([p, p])).sum(),
            bounds=(self.cost, self.a + 2), method="bounded",
            options={"xatol": 1e-10})
        return float(res.x)

    def price_grid(self) -> np.ndarray:
        pn, pm = self.nash_price, self.monopoly_price
        span = self.xi * (pm - pn)
        return np.linspace(pn - span, pm + span, self.n_prices)

    def collusion_index(self, avg_price: float) -> float:
        pn, pm = self.nash_price, self.monopoly_price
        return float((avg_price - pn) / (pm - pn))

    def static_br_on_grid(self) -> np.ndarray:
        """Grid index of the best response to each rival grid price — the
        regulator's memoryless-null benchmark used by the audit."""
        grid = self.price_grid()
        br = np.empty(len(grid), dtype=np.int64)
        for j, pj in enumerate(grid):
            profits = [self.profits(np.array([pi, pj]))[0] for pi in grid]
            br[j] = int(np.argmax(profits))
        return br


if __name__ == "__main__":
    mkt = BertrandMarket()
    pn, pm = mkt.nash_price, mkt.monopoly_price
    print(f"p_N = {pn:.4f}, p_M = {pm:.4f} "
          f"(Calvano et al. benchmarks: ~1.473, ~1.925)")
    print(f"profits at Nash: {mkt.profits(np.array([pn, pn])).round(4)}")
    print(f"profits at monopoly: {mkt.profits(np.array([pm, pm])).round(4)}")
    print(f"grid: {mkt.price_grid().round(3)}")
