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
# Competitive controls: myopic Q-learners (discount 0) converge to static
# Nash play — the population needs non-collusive members for the detection
# (ROC) exhibit, since patient Q-learners collude in nearly every run.
N_CONTROLS = 5


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

    jobs = [(s, 0.95) for s in range(N_SEEDS)] + \
           [(100 + s, 0.0) for s in range(N_CONTROLS)]  # myopic controls
    for seed, delta_rl in jobs:
        key = f"{seed}"
        if key in done:
            continue
        kind = "myopic control" if delta_rl == 0.0 else "patient"
        print(f"seed {seed} ({kind}): training Q pair ({N_STEPS} steps)...")
        pair = train_q_pair(mkt, seed=seed, n_steps=N_STEPS,
                            beta_decay=BETA_DECAY, delta=delta_rl)
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
    patient = [r for r in rows if int(r["seed"]) < 100]
    control = [r for r in rows if int(r["seed"]) >= 100]

    d_p = np.array([float(r["delta"]) for r in patient])
    a_p = np.array([float(r["audit_mean"]) for r in patient])
    rho_intensity = spearman(a_p, d_p)
    print(f"patient n={len(patient)}: Spearman(audit, Delta) = "
          f"{rho_intensity:.2f}")

    det_line = "controls pending"
    auc = np.nan
    if control:
        a_c = np.array([float(r["audit_mean"]) for r in control])
        d_c = np.array([float(r["delta"]) for r in control])
        # Detection: does the audit separate collusive (patient) from
        # competitive (myopic-control) agents? AUC = P(audit_patient >
        # audit_control) over all pairs.
        auc = float(np.mean(a_p[:, None] > a_c[None, :]))
        det_line = (f"controls n={len(control)}: Delta in "
                    f"[{d_c.min():.2f}, {d_c.max():.2f}], audit in "
                    f"[{a_c.min():.3f}, {a_c.max():.3f}]; detection AUC "
                    f"P(audit_collusive > audit_competitive) = {auc:.2f}")
        print(det_line)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(a_p, d_p, s=80, zorder=3, label="patient Q-learners")
    if control:
        ax.scatter(a_c, d_c, s=80, zorder=3, marker="s", color="tab:green",
                   label="myopic controls")
    ax.set_xlabel("internal audit score (no market outcomes used)")
    ax.set_ylabel("realized collusion index $\\Delta$")
    ax.set_title("Audit as detector vs intensity meter "
                 f"(AUC = {auc:.2f}; intensity Spearman = {rho_intensity:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "population.png", dpi=150)

    table = "\n".join(
        f"| {r['seed']} | {r['delta']} | {r['audit_mean']} | "
        f"{r['agree1']}/{r['agree2']} |"
        for r in sorted(rows, key=lambda r: -float(r["delta"])))
    summary = f"""# Population experiment (raw)

n = {len(patient)} patient Q-learning pairs + {len(control)} myopic
controls ({N_STEPS} steps, Calvano et al. parameters).
Spearman(audit, Delta) on patient pairs = {rho_intensity:.2f}
(pre-registered H2 target was > 0.6).
{det_line}
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
