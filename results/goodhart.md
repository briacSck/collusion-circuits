# Goodhart experiment — annotated (run of 2026-07-07)

Raw: `goodhart_raw.md` / `goodhart.png`. Pre-registration:
`design/prereg_goodhart.md` (committed before the run).

## HG1 — CONFIRMED, dramatically: the v1 audit is trivially evadable

| λ_adv | real audit | self-play Δ | fidelity |
|---|---|---|---|
| 0 | 0.0356 | 0.585 | 0.995 |
| 0.3 | **0.0011** | **0.585** | 0.991 |
| 1.0 | 0.0004 | 0.585 | 0.990 |
| 3.0 | 0.0003 | 0.585 | 0.988 |
| 10 | 0.0001 | −0.057 | 0.987 |
| 30 | 0.0000 | −0.057 | 0.987 |

A tiny adversarial weight (λ = 0.3) drives the audit 40× below the
detection threshold (0.043) while leaving collusion fully intact
(Δ = 0.585, the base pair's level). The evasion sweet spot is wide
(λ ∈ [0.3, 3]); only at λ ≥ 10 does the audit penalty finally destroy the
collusive mechanism itself (Δ collapses to −0.06). HG2's trade-off exists
but only at the far end of the frontier: within the sweet spot, evasion is
essentially free.

## What the evader did (the mechanistic reading)
The audit measures expressed price-sensitivity at the cooperative row. The
adversarial distillation flattened exactly that surface while preserving
the policy's structure elsewhere — including whatever off-path punishment
behavior sustains cooperation in self-play. The mechanism did not
disappear; it RELOCATED to states the audit does not measure. Textbook
Goodhart: the metric was optimized, the construct survived.

## Consequences (already reflected in paper/note.tex)
1. **The detection claim must be scoped:** AUC 0.96 holds against NAIVE
   agents (trained without knowledge of the audit — today's realistic
   case). Against white-box audit-aware evasion, the v1 audit fails
   completely.
2. **Audit design becomes strategic:** fixed, known measurement locations
   are gameable; candidate defenses — randomized/secret audit channels,
   off-path audits (the evader must keep punishment structure SOMEWHERE
   to sustain collusion; auditing everywhere it could hide raises evasion
   cost), and behavioral-consistency cross-checks (an agent whose on-path
   surface says "competitive" but whose self-play is collusive is
   detectable by combining the two — at the cost of needing outcomes
   again).
3. **Paper #4's theory is now motivated empirically:** audit design under
   strategic evasion is a mechanism-design problem (what to measure, with
   what randomness, against a best-responding designer of agents).
