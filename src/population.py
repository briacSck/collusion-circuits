"""Population experiment: does the internal audit predict realized collusion?

Pre-registered targets (design/research_design.md §4):
- H1 (anchors) — validated separately by audit.py: grim > TFT > myopic ~
  random. Confirmed 2026-07-07 (0.069 > 0.025 > 0.015 ~ 0.000).
- H2: Spearman(audit score, collusion index Delta) > 0.6 across
  independently trained Q-learning pairs.

Pipeline per seed: train a Q pair (Calvano parameters), record Delta over
the final window, compute greedy self-play visitation, distill BOTH firms'
policies into PricingMLPs (visitation-weighted cloning), audit each network
(NO market outcomes used), store mean pair audit score.

Resumable: results/population_rows.csv. Outputs: population_raw.md + figure.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents import (distill, greedy_selfplay_visitation, mlp_greedy_policy,
                    q_to_policy, train_q_pair)
from audit import audit_scores
from market import BertrandMarket

RESULTS = Path(__file__).resolve().parents[1] / "results"
CSV_PATH = RESULTS / "population_rows.csv"
FIELDS = ["seed", "delta", "audit1", "audit2", "audit_mean",
          "agree1", "agree2"]
N_SEEDS = 20
N_STEPS = 1_000_000
BETA_DECAY = 5e-6


def load_done():
    if not CSV_PATH.exists():
        return {}
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return {row["seed"]: row for row in csv.DictReader(f)}


def append_row(row):
    new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    mkt = BertrandMarket()
    n = mkt.n_prices
    done = load_done()

    for seed in range(N_SEEDS):
        key = f"{seed}"
        if key in done:
            continue
        print(f"seed {seed}: training Q pair ({N_STEPS} steps)...")
        pair = train_q_pair(mkt, seed=seed, n_steps=N_STEPS,
                            beta_decay=BETA_DECAY)
        v1, v2 = greedy_selfplay_visitation(pair.q1, pair.q2, n)

        audits, agrees = [], []
        for q, v, dseed in ((pair.q1, v1, seed), (pair.q2, v2, seed + 10_000)):
            pol = q_to_policy(q)
            net = distill(pol, n, weights=v, seed=dseed)
            # visitation-weighted greedy agreement (fidelity where it matters)
            agree = float(((mlp_greedy_policy(net, n) == pol.argmax(1))
                           * v / v.sum()).sum())
            audits.append(audit_scores(net, mkt)["score"])
            agrees.append(agree)

        append_row(dict(seed=key, delta=f"{pair.collusion_index:.4f}",
                        audit1=f"{audits[0]:.5f}", audit2=f"{audits[1]:.5f}",
                        audit_mean=f"{np.mean(audits):.5f}",
                        agree1=f"{agrees[0]:.3f}", agree2=f"{agrees[1]:.3f}"))
        print(f"  Delta={pair.collusion_index:.3f}, "
              f"audit={np.mean(audits):.4f}, agree={np.mean(agrees):.3f}")

    build_outputs()


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def build_outputs() -> None:
    rows = list(load_done().values())
    if len(rows) < 5:
        print(f"only {len(rows)} rows — waiting for more before outputs")
        return
    delta = np.array([float(r["delta"]) for r in rows])
    audit = np.array([float(r["audit_mean"]) for r in rows])
    rho = spearman(audit, delta)
    print(f"n={len(rows)} pairs: Spearman(audit, Delta) = {rho:.2f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(audit, delta, s=80, zorder=3)
    ax.set_xlabel("internal audit score (no market outcomes used)")
    ax.set_ylabel("realized collusion index $\\Delta$")
    ax.set_title(f"Audit predicts collusion: Spearman = {rho:.2f} "
                 f"(n = {len(rows)} pairs)")
    fig.tight_layout()
    fig.savefig(RESULTS / "population.png", dpi=150)

    table = "\n".join(
        f"| {r['seed']} | {r['delta']} | {r['audit_mean']} | "
        f"{r['agree1']}/{r['agree2']} |"
        for r in sorted(rows, key=lambda r: -float(r["delta"])))
    summary = f"""# Population experiment (raw)

n = {len(rows)} independently trained Q-learning pairs
({N_STEPS} steps, Calvano et al. parameters).
**Spearman(audit score, collusion index) = {rho:.2f}** (H2 target: > 0.6).
Anchor check (H1, audit.py): grim 0.069 > TFT 0.025 > myopic 0.015 ~
random 0.000 — confirmed.

| seed | Delta | audit (mean of pair) | distill agreement (weighted) |
|---|---|---|---|
{table}
"""
    (RESULTS / "population_raw.md").write_text(summary, encoding="utf-8")
    print(f"outputs -> {RESULTS / 'population_raw.md'}")


if __name__ == "__main__":
    main()
