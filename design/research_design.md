# Research design — Collusion Circuits: Ex-Ante Detection of Coordinated
# Pricing Mechanisms Inside Algorithmic Agents
### Paper #3 of the "Microeconomics of Artificial Agents" program

**Status:** design finalized 2026-07-07, before any code. Written to be
executable by a cold-start session. Methodological engine inherited from
paper #1 (`../structural_ground_truth/`, GitHub `briacSck/structural-interp`):
ground-truth lab × agent population × causal conformity audit × outcome
validation.

## 1. Question and pitch

Calvano, Calzolari, Denicolò & Pastorello (AER 2020) showed Q-learning
pricing agents learn supra-competitive prices sustained by reward–punishment
schemes; Fish, Gonczarowski & Shorrer (2024) showed LLM pricing agents do
too. All existing detection is **behavioral and ex post**: infer collusion
from market outcomes after harm occurs. This paper asks the regulator's
dream question:

> Can a causal audit of a pricing algorithm's INTERNALS — run on
> in-distribution data, before or without observing market outcomes —
> detect the mechanism that sustains collusion?

The audit concept transfers directly from paper #1: define the **null
mechanism class** a regulator would accept (memoryless/competitive best
response to the current state), and measure the causal conformity of the
agent's internals to that class. The collusion-sustaining channel —
responsiveness to *rival price history* beyond current-period best response,
i.e. the trigger/punishment structure — is exactly an out-of-class mechanism
channel, and paper #1 showed such channels are auditable by directional
patching with dispersion statistics.

## 2. The laboratory

### 2.1 Market game
Repeated Bertrand duopoly with logit demand (the Calvano et al.
parameterization to stay comparable): firms i ∈ {1,2}, discrete price grid
(15 points spanning below-competitive to above-monopoly), per-period profits
π_i(p_i, p_j), discount δ = 0.95. Known benchmarks: competitive
(Bertrand-Nash) price p^N and monopoly price p^M; the collusion index
Δ = (p̄ − p^N)/(p^M − p^N) is the standard outcome measure.

### 2.2 Agents
Small NN policies (2-hidden-layer MLPs, same scale as paper #1 — laptop
CPU) mapping a state to a price distribution. State = last k periods of
both prices (k = 1 as baseline, matching Calvano's memory-one). Training:
independent REINFORCE or DQN, self-play pairs, long horizon, ε-decay.
Expected from the literature: a fraction of trained pairs converges to
supra-competitive pricing with punishment dynamics; others converge near
Nash. THIS HETEROGENEITY IS THE IDENTIFICATION STRATEGY: the population of
trained pairs spans a range of collusion indices Δ, and the audit must
rank-predict Δ.

Also include DESIGNED agents as anchors (paper #1's zoo trick, ground truth
by construction): a hard-coded grim-trigger network, a tit-for-tat-ish
network, a myopic best-response network, and a random-pricing network. These
give the audit a known ordering to be validated against before the learned
population is scored.

### 2.3 The audit (the paper's contribution)
Null class: price depends only on payoff-relevant current state (with
memory-one state, the null is "responds to rival's price only as much as
static best response does"). Audit channels, by directional/amnesic
patching (port `probes.py`, `leace.py`, dispersion machinery from paper #1):
1. **History-sensitivity channel:** causal effect of patching the rival-
   price-history component of hidden activations, compared against the
   static-best-response benchmark computed from the known demand system.
   Excess sensitivity = trigger structure.
2. **Asymmetry channel:** punishment schemes respond asymmetrically to
   rival undercutting vs overpricing — dispersion/asymmetry of the patched
   response across the price grid (the c-conformity construction, with the
   insensitivity gate).
Conformity-to-null score per agent; validated first on the designed anchors
(grim-trigger must score maximally non-conforming; myopic BR conforming).

## 3. Headline experiment
Train N pairs (target N ≥ 50; embarrassingly parallel, resumable CSV per
pair, crc32 seeds). For each pair: (i) realized collusion index Δ over the
last T periods; (ii) audit scores computed WITHOUT market outcomes (weights
+ in-distribution activations only). Headline: Spearman(audit, Δ) across
the population + ROC curve for "Δ > 0.3" detection. Secondary: audit the
agents mid-training (does the trigger channel form BEFORE prices rise? —
the truly ex-ante claim).

## 4. Pre-registered hypotheses (to commit before running)
- **H1 (anchor validity):** designed agents are ordered correctly by the
  audit (grim > TFT > myopic ≈ random).
- **H2 (population prediction):** Spearman(audit, Δ) > 0.6 across learned
  pairs; history-sensitivity channel dominates.
- **H3 (temporal precedence):** in pairs that end up collusive, the audit
  score rises before the price series reaches supra-competitive levels
  (audit leads outcome). This is the strongest, riskiest claim — if it
  fails, report it; H2 alone carries the paper.

## 5. Implementation plan
1. `src/market.py`: logit-demand Bertrand game, Nash/monopoly benchmarks
   (closed-form or fixed point), profit matrices. Unit tests: p^N < p^M,
   best-response function sanity.
2. `src/agents.py`: MLP policy, REINFORCE/DQN training loop, designed
   anchor agents (grim-trigger etc. as hand-set networks).
3. `src/audit.py`: ports of directional patching / LEACE / dispersion-with-
   gate from paper #1, adapted to the price-history channels.
4. `src/population.py`: resumable N-pair training + audit + Δ, CSV.
5. Figures: audit-vs-Δ scatter; ROC; mid-training trajectories (H3).
6. Note drafting on the paper #1 skeleton.

## 6. Policy tie-in (DG COMP)
The ex-officio Cartels data team's mandate is proactive detection via
statistical screening. This design is the algorithmic-agent version of a
screen: mechanism-level, ex ante, no market data required. Use in the
motivation letter (Géza Sápi; Blue Book registration opens 2026-07-15):
one paragraph on moving from outcome-based screens to mechanism audits of
pricing algorithms, citing Calvano et al. and Fish et al. — the exact
literature the posting asks trainees to monitor.

## 7. Risks and mitigations
- **Collusion may not emerge reliably with NN + REINFORCE** (most Calvano
  results use tabular Q-learning). Mitigations: use DQN with small nets;
  memory-one state keeps it close to tabular; worst case, distill tabular
  Q-learners into networks (behavior cloning — paper #1's technique) and
  audit the distilled nets. The anchors guarantee the audit exhibit exists
  regardless.
- **Scoop risk** (Fish–Gonczarowski–Shorrer orbit, interp community): the
  arXiv watch is running; the differentiator to protect is the
  conformity-audit framing + anchor-validated population design.
- **Compute:** hundreds of small training runs; still CPU-feasible but
  budget hours, use the resumable pattern from day one.

## 8. Must-read before executing
Calvano et al. (2020, AER 110(10)); Fish, Gonczarowski & Shorrer (2024,
arXiv "Algorithmic Collusion by Large Language Models"); Klein (2021, RAND,
sequential pricing collusion); Assad et al. (2024, JPE:Micro, German retail
gasoline — empirical algorithmic pricing); Harrington (2018, "Developing
Competition Law for Collusion by Autonomous Artificial Agents", J. Comp. L.
& Econ.) for the legal framing; paper #1's audit sections. Verify all with
a feynman sweep before citing (only Calvano and Fish et al. were verified in
prior sweeps).
