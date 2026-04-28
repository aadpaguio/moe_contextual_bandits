The flow is mostly good but the framing has a problem worth fixing before you start writing. Let me work through it.

## What's working

- Theory → methodology → results → discussion is the right scaffold.
- Including baselines, seeds, and the alpha-sweep is methodologically clean.
- Future work pointing at GLM-ES and NeuralUCB is well-aimed (more expressive reward models that match your finding that linear-in-x is the bottleneck).

## What needs fixing

**1. The presentation/report flow is conflating two audiences.** Per the instructions you have 10 minutes max for presentation but the report has different sections (Introduction, Background, Problem Setup, Results, Experiments, Open Problems). Your bullet list reads as a presentation outline, not a report outline. Map them separately. The report has more room and should follow the rubric's section names; the presentation extracts highlights.

**2. "Theory Proof" is vague and risky.** What proof? You haven't done original theory in this chat. If you mean "we walk through Theorem E.1 from Lattimore-Szepesvári-Weisz to motivate the misspecification framing," say that explicitly. The instructions say "Be detailed! Provide proofs" — they want either your own derivation or a clean re-exposition of someone else's. Aidan was supposed to handle theory; coordinate on what proof goes in. If it's the LinUCB regret bound from Chu et al., own that — say "we re-derive [or carefully expose] the LinUCB confidence interval and regret bound, with attention to the misspecified case."

**3. You've cut independent experts for the talk — should you cut them from the report?** Different question. The report has more space. I'd keep one paragraph + one figure on independent experts in the report as motivation for the joint-training setup. It supplies the "why does smoothing matter" backstory that the theorem is trying to capture. Don't make it a results section, make it a motivation/methodology footnote.

**4. The "regret is generally linear" line should not be a section header.** It's a finding worth discussing, but framing it as "regret is linear" without context will read like "our methods don't work." Frame it as: "all policies show approximately linear cumulative regret at T=10k, indicating none have reached their asymptotic regime; the relevant comparison is per-step regret, where LinUCB pays a misspecification cost ε that softmax does not." That's the same finding stated correctly.

**5. Missing: separate the two softmax routers.** Online softmax and offline
supervised softmax answer different questions. Online softmax is a fair bandit
baseline because it only observes chosen-arm rewards. Offline supervised
softmax is a full-information learned-router reference trained from best-arm
labels on expert-training contexts. Your results now say: LinUCB is roughly
neck-and-neck with online softmax, and consistently beats offline supervised
softmax. That is the headline comparison.

**6. Future work is thin.** GLM-ES and NeuralUCB are good but generic. Add specific ones tied to your actual findings:
- "Whether longer horizons T ≫ 10k separate LinUCB and online softmax, which
  are close at T=10k."
- "Whether per-d α tuning changes the near-tie between LinUCB and online
  softmax."
- "Whether n_train scaling with d (we observed val acc degradation at d=64 with fixed n_train=8000) was masking the effect."

These are *your* open questions, not generic next-steps.

## Revised flow for the report

**Introduction**
- MoE routing as contextual bandit. The "bandit-as-cheap-adapter" pitch.
- Concrete claim: bandit routers as adapters over frozen jointly-trained MoE experts.
- Why this matters: avoids retraining experts for new tasks.

**Background**
- Standard linear contextual bandit setup, LinUCB (Li et al. 2010).
- Misspecification: Lattimore-Szepesvári-Weisz 2020 Theorem E.1, which says LinUCB regret is √T·d log T + ε·T·√(d log T).
- Prior work: nothing has cast MoE routing as a bandit problem this way (verify this!).

**Problem Setup**
- Synthetic Gaussian clusters in R^d, K experts, soft-routed joint training.
- Reward: log-prob of true label by chosen expert.
- Regret definition with respect to oracle arm per context.
- Stationarity: experts frozen after joint training, so per-step reward is i.i.d.

**Results / Analysis**
- LinUCB confidence interval and regret bound (the proof you'd have Aidan write up). Walk through the key step: how the misspecification term enters.
- Implementation details for joint training (Switch-style load balancing, α_load, soft routing).
- Why we use joint training: brief mention of independent training as a counterfactual that produces large ε (one figure: reward heatmap), motivating the joint-training approach.

**Experiments**
- Data generation: cluster centers via `ortho_group` (after the fix you should make), noise, T=10k, n_train=8000.
- Baselines: uniform, ε-greedy, LinUCB, online softmax, offline supervised
  softmax, oracle.
- Online softmax is the fair learned bandit router: it learns only from
  observed rewards.
- Offline supervised softmax is a full-information learned-router reference:
  train on `X_train` using frozen-expert best-arm labels, then evaluate on
  `X_bandit`. Do not call this an online method or an upper bound.
- 3 seeds per cell, mean curves shown.
- d-sweep results: regret vs. d for each policy. Note the linear-regret pattern and explain.
- α-sweep results across d. Show how LinUCB performance changes with α and
  what α you select for the main runs.
- Gap plots:
  - `LinUCB - online softmax`: near zero with sign changes across d, showing
    the two online methods are competitive.
  - `LinUCB - offline supervised softmax`: consistently negative, showing
    LinUCB beats the full-information offline classifier in this setup.

**Discussion**
- Joint training successfully reduces misspecification (RMSE ε drops 49% vs. independent at d=4).
- LinUCB and online softmax are closely matched at T=10k, so the fair online
  comparison does not clearly crown either method across all d.
- LinUCB consistently outperforming offline supervised softmax suggests that
  static best-arm classification is not sufficient for this routing problem,
  even when trained with more information than an online bandit policy gets.
- The bottleneck shifts from misspecification to the interaction between
  finite-horizon exploration, reward noise, and how well each router can adapt
  to frozen expert rewards.
- Tradeoff: joint training also reduces contextual richness — at low sep, ε-greedy nearly matches LinUCB.

**Open problems / future directions**
- Longer-horizon experiments to test sub-linear LinUCB regret asymptotic.
- NeuralUCB / GLM-UCB to relax linear-in-x assumption — does it separate the
  online methods from online softmax?
- Whether α tuning per d shifts the d-sweep conclusion.
- n_train scaling with d to control for expert quality.
- Real MoE deployment: do these synthetic findings transfer to learned embeddings?

## Revised flow for the 10-min talk

You don't need to mirror the report. Optimize for one clear takeaway:

1. **Hook (30 sec):** "MoE routing is a bandit problem. Can we get away with a cheap online adapter instead of supervised training?"
2. **Setup (1 min):** Synthetic clusters, joint training, frozen experts,
   LinUCB vs two softmax routers.
3. **Theorem E.1 (1 min):** Misspecification penalty εT√(d log T). The thing to fear.
4. **Joint training fixes ε (2 min):** Show RMSE ε ≈ 1.5 across d. Misspecification controlled.
5. **d-sweep result (3 min):** LinUCB is close to online softmax and beats
   offline supervised softmax. Use the two gap plots to show the two claims.
6. **Interpretation (1 min):** Online reward adaptation is competitive with
   learned stochastic routing, while offline best-arm classification is not
   enough to solve routing in this setting.
7. **Open questions (1 min):** Longer T, NeuralUCB, etc.
8. **Q&A buffer (30 sec).**

Independent experts only mentioned if asked.

## One more thing to verify before writing

You said Aidan handles theory and you handle implementation. The "Theory Proof" line in your outline needs to actually map to a deliverable he's producing. Make sure that's coordinated — don't end up with the report missing its theory section because both of you assumed the other was writing it. Sync with him this week, agree on what theorem/proof goes in, and confirm before you start drafting the report.