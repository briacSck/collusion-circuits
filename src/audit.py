"""Conformity audit of pricing networks against the competitive null.

Null mechanism class (the regulator's benchmark): memoryless static best
response — the agent's price responds to the rival's last price exactly as
the one-shot best-response mapping does, no more.

Channels (v1, memory-one caveat documented in the design doc):
1. EXCESS SENSITIVITY: causal response of the network's expected price to
   adjacent rival-price changes (activation patching at layer 0, forward
   through the rest), minus the static-BR response, positive part.
   Punishment/trigger structures respond MORE steeply than BR (grim's cliff,
   TFT's slope-1 matching); competitive agents ~ 0; unresponsive agents
   clip to 0 (the insensitivity-gate logic of paper #1: under-responding is
   not collusion).
2. UNDERCUT ASYMMETRY: punishment schemes react more to rival undercutting
   than to rival price increases; positive part of the difference.

Audit score = excess + asymmetry, computed on in-distribution activations
only (no market outcomes, no simulation of play).
"""

import numpy as np
import torch

from agents import PricingMLP, state_features, state_index
from market import BertrandMarket


def _forward_from(net: PricingMLP, layer: int, activation: torch.Tensor):
    h = activation
    for module in net.net[2 * layer + 2:]:
        h = module(h)
    return h


def expected_price_from_acts(net: PricingMLP, layer: int,
                             acts: torch.Tensor,
                             grid: np.ndarray) -> float:
    logits = _forward_from(net, layer, acts)
    probs = torch.softmax(logits, dim=-1).numpy()
    return float(probs @ grid)


def audit_scores(net: PricingMLP, mkt: BertrandMarket,
                 layer: int = 0) -> dict:
    grid = mkt.price_grid()
    n = mkt.n_prices
    br = mkt.static_br_on_grid()
    pm_idx = int(np.argmin(np.abs(grid - mkt.monopoly_price)))

    feats = state_features(n)
    with torch.no_grad():
        acts = net.hidden_activations(feats)[layer]

        # Expected-price response profile at the cooperative on-path own
        # price (where the punishment threat lives), via layer-0 patching.
        own = pm_idx
        e_p = np.array([
            expected_price_from_acts(
                net, layer, acts[state_index(own, j, n)], grid)
            for j in range(n)])

    resp = np.abs(np.diff(e_p))                    # |dE[p]| per rival step
    br_resp = np.abs(np.diff(grid[br]))            # BR benchmark response
    excess = float(np.clip(resp - br_resp, 0, None).mean())

    # Asymmetry around the cooperative price: response to undercutting
    # (rival below pm) vs to increases (at/above pm).
    down = resp[:pm_idx].mean() if pm_idx >= 1 else 0.0
    up = resp[pm_idx:].mean() if pm_idx < len(resp) else 0.0
    asym = float(max(down - up, 0.0))

    return {"excess": excess, "asym": asym, "score": excess + asym}


if __name__ == "__main__":
    from agents import ANCHORS, anchor_policy, distill

    mkt = BertrandMarket()
    print("auditing designed anchors (H1: grim > TFT > myopic ~ random)...")
    for name in ANCHORS:
        net = distill(anchor_policy(name, mkt), mkt.n_prices)
        s = audit_scores(net, mkt)
        print(f"{name:>13}: excess={s['excess']:.4f}, asym={s['asym']:.4f}, "
              f"score={s['score']:.4f}")
