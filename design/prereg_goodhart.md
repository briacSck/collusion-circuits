# Pre-registration — Goodharting the audit (robustness section of THIS paper)

**Date: 2026-07-07, committed BEFORE running the experiment.**
Placement decision (user): this experiment is the robustness/threat-model
section of the collusion-circuits paper; the theory paper (#4) will cite it.

## Question
Can a pricing agent be constructed to PASS the conformity audit while
remaining collusive? If yes, the paper's policy claim ("authorities could
screen mechanisms ex ante") needs a caveat about strategic evasion BEFORE
circulation; if no (within the attack class), the claim strengthens.

## Attack model (the strongest reasonable one)
White-box, audit-aware distillation: the evader knows the audit exactly and
trains against it. We take a trained collusive Q-pair (seed 0 of the
population, Delta = 0.56) and re-distill BOTH firms' policies into
PricingMLPs with the loss

    CE(policy target, net) + lambda_adv * AuditProxy(net)

where AuditProxy is the differentiable version of the audit score (expected
price via softmax at the cooperative row; excess = relu(|dE[p]| - |dBR|)
mean; asymmetry = relu(down - up); proxy = excess + asym). Sweep
lambda_adv over {0, 0.3, 1, 3, 10, 30}, seeds fixed.

## Metrics per lambda_adv
1. Audit score of each distilled net (the REAL audit, patching-based, as in
   audit.py — not the proxy).
2. Realized collusion: greedy self-play of the two distilled nets for 5000
   periods -> Delta_adv.
3. Behavioral fidelity: visitation-weighted greedy agreement with the
   original collusive policy.

Detection threshold: the maximum audit score among the population's myopic
controls (0.043). "Passing the audit" = mean pair audit below it.

## Pre-registered hypotheses
- **HG1 (evasion exists):** for some lambda_adv, the pair's audit score
  falls below the detection threshold while Delta_adv remains > 0.3
  (comparable to the population's collusive range). If confirmed: audits
  are Goodhart-able under white-box attack; the paper's policy section adds
  strategic-evasion caveats and the audit-design implication (secret or
  randomized audit channels; off-path audits harder to game because play
  never expresses them).
- **HG2 (evasion is costly):** the evasion frontier is not free — audit
  score and Delta_adv trade off: as lambda_adv rises, either Delta falls
  (collusion sacrificed) or fidelity falls before the threshold is reached.
  We report the frontier either way.

Both outcomes are informative and will be reported as they come out; no
post-hoc exclusion of lambda values.

## Analysis commitments
Single run per lambda (deterministic seeds); frontier figure (audit score
vs Delta_adv, threshold line, lambda annotated); results into
results/goodhart_raw.md, annotated separately.
