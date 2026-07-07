# Population experiment — annotated (run of 2026-07-07, n = 20 + 5)

Raw: `population_raw.md` / `population_rows.csv` (deduplicated after a
concurrent-writer incident — identical rows from deterministic seeds, no
corruption; single-writer discipline noted for future runs).

## Verdict against the pre-registration

**H1 (anchors) — CONFIRMED.** grim 0.069 > TFT 0.025 > myopic 0.015 ≈
random 0.000: the audit orders punishment structure correctly on agents
whose ground truth is known by construction.

**H2 as pre-registered (intensity meter) — REJECTED.**
Spearman(audit, Δ) = **−0.36** across the 20 patient Q-learning pairs
(all collusive, Δ ∈ [0.28, 0.72]). Reported as pre-registered.

**Detection (the regulator's actual question) — STRONG.**
Adding 5 myopic-control pairs (δ = 0 Q-learners, Δ ∈ [0.02, 0.20]):
**AUC = 0.96** — the audit ranks a randomly drawn collusive agent above a
randomly drawn competitive one 96% of the time, using only weights and
in-distribution activations, no market outcomes.

## Why the intensity meter fails (mechanism, for the note)
The v1 audit measures the *expressed* trigger structure in the response
profile at the cooperative row. Highly collusive pairs sustain cooperation
with credible threats that are rarely exercised: their on-path policies are
smooth and stable, so measured excess sensitivity is LOW — while moderately
collusive pairs cycle through punishment episodes, leaving jagged expressed
profiles. Threat depth is an off-path object; an intensity meter needs
off-path structure (the Q-table's punishment valleys, multi-row response
surfaces), not on-path sensitivity. This mirrors paper #1's lesson: each
audit sees its own channel.

## Paper framing
The deliverable a competition authority needs is the screen, and the screen
works: **collusive vs competitive mechanism, AUC 0.96, ex ante**. The
intensity anti-correlation is an honest finding with a clean mechanistic
explanation and an obvious follow-up (off-path audits; H3 temporal
precedence — does the trigger circuit form before prices rise? — remains
to run, mid-training checkpoints required).
