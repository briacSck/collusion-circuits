"""Goodhart experiment: adversarial distillation against the audit.

Pre-registered: design/prereg_goodhart.md (committed before this run).
Attack: white-box audit-aware distillation of a collusive pair —

    loss = CE(target policy) + lambda_adv * AuditProxy(net)

AuditProxy is the differentiable analog of the audit score (softmax expected
price at the cooperative row; excess over static BR + undercut asymmetry).
Metrics per lambda_adv: REAL patching audit score, self-play collusion
index of the two adversarial nets, visitation-weighted fidelity.

Outputs: results/goodhart_raw.md + results/goodhart.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from agents import (PricingMLP, greedy_selfplay_visitation, q_to_policy,
                    state_features, state_index, train_q_pair)
from audit import audit_scores
from market import BertrandMarket

RESULTS = Path(__file__).resolve().parents[1] / "results"
LAMBDAS = (0.0, 0.3, 1.0, 3.0, 10.0, 30.0)
DETECTION_THRESHOLD = 0.043  # max audit among population myopic controls


def audit_proxy(net: PricingMLP, mkt: BertrandMarket,
                feats: torch.Tensor, grid_t: torch.Tensor,
                br_resp: torch.Tensor, pm_idx: int) -> torch.Tensor:
    """Differentiable audit score at the cooperative own-price row."""
    n = mkt.n_prices
    rows = torch.tensor([state_index(pm_idx, j, n) for j in range(n)])
    logits = net(feats[rows])
    e_p = torch.softmax(logits, dim=-1) @ grid_t          # E[p](rival j)
    resp = (e_p[1:] - e_p[:-1]).abs()
    excess = torch.relu(resp - br_resp).mean()
    down = resp[:pm_idx].mean() if pm_idx >= 1 else torch.tensor(0.0)
    up = resp[pm_idx:].mean() if pm_idx < len(resp) else torch.tensor(0.0)
    return excess + torch.relu(down - up)


def adversarial_distill(policy: np.ndarray, mkt: BertrandMarket,
                        lambda_adv: float, weights: np.ndarray,
                        n_epochs: int = 8000, lr: float = 2e-3,
                        seed: int = 0) -> PricingMLP:
    torch.manual_seed(seed)
    n = mkt.n_prices
    feats = state_features(n)
    target = torch.tensor(policy, dtype=torch.float32)
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()

    grid = mkt.price_grid()
    grid_t = torch.tensor(grid, dtype=torch.float32)
    br_resp = torch.tensor(np.abs(np.diff(grid[mkt.static_br_on_grid()])),
                           dtype=torch.float32)
    pm_idx = int(np.argmin(np.abs(grid - mkt.monopoly_price)))

    net = PricingMLP(n, hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    logsoft = nn.LogSoftmax(dim=-1)
    for _ in range(n_epochs):
        opt.zero_grad()
        ce = -((target * logsoft(net(feats))).sum(dim=-1) * w).sum()
        loss = ce + lambda_adv * audit_proxy(net, mkt, feats, grid_t,
                                             br_resp, pm_idx)
        loss.backward()
        opt.step()
    return net


def mlp_selfplay_delta(net1: PricingMLP, net2: PricingMLP,
                       mkt: BertrandMarket, n_periods: int = 5000) -> float:
    """Collusion index of greedy self-play between the two networks."""
    n = mkt.n_prices
    grid = mkt.price_grid()
    feats = state_features(n)
    with torch.no_grad():
        a1_map = net1(feats).argmax(dim=-1).numpy()
        a2_map = net2(feats).argmax(dim=-1).numpy()
    a1, a2 = n // 2, n // 2
    total = 0.0
    for _ in range(n_periods):
        s1, s2 = state_index(a1, a2, n), state_index(a2, a1, n)
        a1, a2 = int(a1_map[s1]), int(a2_map[s2])
        total += (grid[a1] + grid[a2]) / 2
    return mkt.collusion_index(total / n_periods)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    mkt = BertrandMarket()
    n = mkt.n_prices
    print("training the base collusive pair (seed 0, as in the population)...")
    pair = train_q_pair(mkt, seed=0, n_steps=1_000_000, beta_decay=5e-6)
    v1, v2 = greedy_selfplay_visitation(pair.q1, pair.q2, n)
    pol1, pol2 = q_to_policy(pair.q1), q_to_policy(pair.q2)
    print(f"base pair Delta = {pair.collusion_index:.3f}")

    rows = []
    for lam in LAMBDAS:
        nets, audits, fids = [], [], []
        for pol, v, dseed in ((pol1, v1, 0), (pol2, v2, 10_000)):
            net = adversarial_distill(pol, mkt, lam, v, seed=dseed)
            nets.append(net)
            audits.append(audit_scores(net, mkt)["score"])
            with torch.no_grad():
                greedy = net(state_features(n)).argmax(dim=-1).numpy()
            fids.append(float(((greedy == pol.argmax(1)) * v / v.sum()).sum()))
        delta_adv = mlp_selfplay_delta(nets[0], nets[1], mkt)
        rows.append(dict(lam=lam, audit=float(np.mean(audits)),
                         delta=delta_adv, fidelity=float(np.mean(fids))))
        print(f"lambda_adv={lam}: audit={np.mean(audits):.4f}, "
              f"Delta={delta_adv:.3f}, fidelity={np.mean(fids):.3f}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot([r["audit"] for r in rows], [r["delta"] for r in rows], "o-")
    for r in rows:
        ax.annotate(f"$\\lambda$={r['lam']}", (r["audit"], r["delta"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax.axvline(DETECTION_THRESHOLD, color="red", ls="--",
               label=f"detection threshold ({DETECTION_THRESHOLD})")
    ax.set_xlabel("real audit score (patching-based)")
    ax.set_ylabel("self-play collusion index $\\Delta$")
    ax.set_title("The evasion frontier: audit score vs collusion under "
                 "white-box attack")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "goodhart.png", dpi=150)

    table = "\n".join(
        f"| {r['lam']} | {r['audit']:.4f} | {r['delta']:.3f} | "
        f"{r['fidelity']:.3f} |" for r in rows)
    evaded = [r for r in rows
              if r["audit"] < DETECTION_THRESHOLD and r["delta"] > 0.3]
    verdict = ("HG1: evasion EXISTS at lambda in "
               + str([r["lam"] for r in evaded])) if evaded else \
        "HG1 not achieved in this sweep (see frontier for HG2 trade-off)"
    summary = f"""# Goodhart experiment (raw)

Base pair: seed 0, Delta = {pair.collusion_index:.3f}. Detection threshold
= {DETECTION_THRESHOLD} (max myopic-control audit). {verdict}.

| lambda_adv | real audit | self-play Delta | fidelity |
|---|---|---|---|
{table}
"""
    (RESULTS / "goodhart_raw.md").write_text(summary, encoding="utf-8")
    print(f"outputs -> {RESULTS / 'goodhart_raw.md'}")


if __name__ == "__main__":
    main()
