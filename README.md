# Collusion Circuits
### Paper #3 of the "Microeconomics of Artificial Agents" program

**Question.** Can a causal audit of a pricing algorithm's *internals* — run
on in-distribution data, without observing market outcomes — detect the
punishment/trigger mechanism that sustains algorithmic collusion?

**Approach.** The conformity-audit engine of paper #1
([briacSck/structural-interp](https://github.com/briacSck/structural-interp))
applied to antitrust: the regulator's null mechanism class is memoryless
best response; excess causal sensitivity to *rival price history* — measured
by directional patching of hidden activations — is the collusion channel.
Population design: many independently trained NN pricing pairs (repeated
Bertrand, Calvano et al. 2020 benchmark) spanning a range of collusion
indices, plus hand-designed anchor agents (grim trigger, tit-for-tat, myopic
best response) whose ground truth is known by construction. Headline:
Spearman(audit score, realized collusion index), and whether the audit
*leads* the price series in time.

**Policy hook.** Ex-ante mechanism screening for pricing algorithms — the
algorithmic-agent version of DG COMP's proactive cartel-detection mandate.

**Status.** Full research design (lab, audit channels, pre-registration
targets H1–H3, implementation plan, risks, reading list):
[`design/research_design.md`](design/research_design.md). No code yet —
next step is `src/market.py` per the implementation plan.
