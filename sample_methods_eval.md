Below is a draft you can adapt directly into the report. I’m writing it as a **story-driven Experiments / Methodology / Results section**, with figure callouts embedded where they should appear.

---

# Experiments and Methodology

## Experimental Motivation: Why Joint Training?

Our goal is to study whether Mixture-of-Experts routing can be treated as a contextual bandit problem. In this framing, each expert is an arm, the token or input embedding is the context, and the router’s objective is to choose the expert with the highest reward. We evaluate this question in a controlled synthetic setting where the true expert rewards can be measured after training.

A key design choice is how the experts themselves are trained. We initially considered two regimes: **independent training** and **joint training**. In the independent regime, each expert is trained primarily on one partition of the data, so each expert specializes strongly in one cluster. This creates a reward surface with sharp “spike-and-plateau” structure: one expert performs very well on its own cluster, while the others perform similarly poorly. This setting is useful as a diagnostic, but it is less representative of modern MoE systems, where experts and routers are typically trained together.

In the joint regime, all experts are trained as part of a single soft-routed MoE model. This produces a smoother reward landscape: experts still differ, but the reward gaps are less extreme. This matters for LinUCB because the standard linear contextual bandit analysis assumes that rewards are approximately linear in the context. If the expert reward functions are highly discontinuous or cluster-specific, then the linear model is badly misspecified. Joint training reduces this misspecification and makes the bandit formulation more plausible.

**Figure to show in main text or appendix:**
Use `epsilon_rmse_comparison.png` as the main supporting figure if space allows. Put `reward_heatmap_independent.png` and `reward_heatmap_joint.png` in the appendix. The heatmaps are useful but more motivational than central.

**Suggested figure caption:**

> **Figure X. Independent vs. joint expert training.** Independent training produces a sharper cluster-specialized reward surface, while joint training produces smoother expert rewards and lower linear approximation error. Because LinUCB assumes approximately linear rewards in context, the joint setting provides a more realistic and analytically appropriate evaluation regime.

This experimental split is also important because contamination is only meaningful in the independent regime. In joint training, all data are pooled across clusters, so contamination has no mechanism to affect the learned experts. Therefore, the main results focus only on joint training and do not plot joint regret as a function of contamination, following the packet design constraint. 

---

## Main Experimental Setup

After the motivational independent-vs-joint comparison, all main experiments use the **joint-trained expert regime**. We generate synthetic Gaussian-cluster data in dimension
$$d \in {2,4,8,16,32,64}$$
with (K=4) experts and a fixed cluster separation. For each dimension, we train a joint MoE model, freeze the experts, and then evaluate different routing policies on a bandit stream of (T=10{,}000) examples.

At each timestep (t), the router observes a context (x_t \in \mathbb{R}^d), selects an expert (a_t), and receives the reward associated with that chosen expert. Rewards are defined using the frozen expert’s prediction quality on the true label. Regret is measured relative to an oracle router that always selects the best expert for that context:

$$
r_t^\star = \max_a r_t(a)
$$

and

$$
\text{Regret}(T) = \sum_{t=1}^T \left(r_t^\star - r_t(a_t)\right).
$$

This setup lets us evaluate routing policies by the metric that matters for bandit routing: **cumulative regret**, not simply whether the policy recovers the exact oracle arm.

---

## Policies Compared

We compare LinUCB against several baselines. These baselines are designed to separate three possible explanations for performance: random routing, simple online exploration, learned online adaptation, and offline supervised routing.

### Uniform Router

The uniform router chooses each expert with equal probability. This provides a lower-bound baseline for routing quality. Since it does not use context or rewards, any reasonable contextual policy should outperform it.

### Epsilon-Greedy Router

The epsilon-greedy router uses a decaying exploration schedule,

$$
\epsilon_t = \min(1, c/t),
$$

with (c=50). It serves as a simple online bandit baseline that can adapt from reward feedback but does not explicitly model context in the same way as LinUCB.

### LinUCB Router

LinUCB maintains a separate linear reward model for each expert. At each timestep, it chooses the arm with the highest upper confidence bound:

$$
\hat{r}_a(x_t) + \alpha \cdot \text{uncertainty}_a(x_t).
$$

The parameter (\alpha) controls the exploration-exploitation tradeoff. We sweep

$$
\alpha \in {0.5, 1.0, 2.0, 4.0, 8.0}
$$

for each dimension and select the best (\alpha) by mean cumulative regret. This avoids relying on a single arbitrary exploration parameter.

### Online Softmax Router

The online softmax router is the fairest learned online comparator. It uses a stochastic softmax policy and updates online from observed feedback during the evaluation stream. Unlike the offline supervised router, it is not frozen before evaluation; like LinUCB, it adapts as data arrives.

This baseline tests whether a flexible learned router can match or beat LinUCB under the same online routing setting. In the report packet, this comparison is explicitly treated as the fair online-method comparison. 

**Important naming note:**
If the implementation uses observed rewards only, call this `online_softmax_bandit` or `online_softmax_pg`, not `online_softmax_best_arm`. The name `best_arm` implies online oracle labels, which would change the interpretation.

### Offline Supervised Softmax Router

The offline softmax router is a supervised learned-router reference. After the experts are frozen, we compute the best expert on an offline training set and train a small MLP to predict the best-arm label. The router is then frozen and evaluated on the bandit stream.

This approximates the standard supervised MoE routing workflow: train the router once, then deploy it as a fixed policy. However, it receives stronger offline information than LinUCB because it is trained on best-arm labels rather than chosen-arm rewards. The report packet correctly emphasizes that this baseline should be interpreted as a **full-information offline learned-router reference**, not as an online bandit method or an upper bound. 

This baseline addresses a key question: is best-arm classification enough to solve routing? If the offline softmax achieves high best-arm accuracy but still suffers higher regret, then classification accuracy is not fully aligned with routing quality.

### Oracle Router

The oracle router always chooses the best expert for each context and therefore has zero regret. It is not a deployable policy, but it defines the benchmark against which regret is computed.

---

# Results

## Result 1: Joint Training Reduces Linear Misspecification

The first experimental result is motivational: joint training substantially reduces the linear approximation error of expert rewards compared with independent training. This matters because LinUCB assumes that each arm’s reward can be approximated by a linear function of the context. When independent training creates highly cluster-specialized experts, the reward surface is less linear and more discontinuous. Joint training smooths the reward landscape and makes a linear contextual bandit model more appropriate.

**Figure to show:**
Main text: `epsilon_rmse_comparison.png`
Appendix: `reward_heatmap_independent.png`, `reward_heatmap_joint.png`

**Suggested prose:**

The independent expert regime produces sharper specialization, but this also increases misspecification for a linear bandit router. In contrast, joint training reduces the RMSE of the linear reward approximation. This supports our decision to use independent training only as a motivational diagnostic, while using joint-trained experts for the main d-sweep.

This result does not mean joint training eliminates misspecification. Rather, it reduces it enough that LinUCB becomes a plausible router. The remaining approximation error is still important and helps explain why cumulative regret remains approximately linear over the evaluated horizon.

---

## Result 2: LinUCB Is Competitive with Online Softmax Across Dimensions

The main d-sweep compares routing policies across dimensions
(d \in {2,4,8,16,32,64}). The headline finding is that LinUCB is highly competitive with the online softmax router. The gap between the two online methods is small and changes sign across dimensions.

**Figure to show:**
Use `regret_vs_d.png` first. Then use `linucb_minus_online_softmax_vs_d.png` immediately after or as a supporting figure.

**Suggested figure order:**

1. `regret_vs_d.png` — headline plot showing all policies.
2. `linucb_minus_online_softmax_vs_d.png` — focused comparison between the two online adaptive routers.
3. `linucb_minus_offline_softmax_vs_d.png` — focused comparison against offline supervised softmax.

**Suggested prose:**

Across the joint-trained d-sweep, LinUCB achieves the lowest mean cumulative regret overall, but its advantage over online softmax is small. This suggests that the fair online comparison does not produce a clear domination result. Instead, LinUCB and online softmax should be interpreted as closely matched online adaptive routers.

This is still a positive result for LinUCB. Online softmax is a more flexible learned policy, while LinUCB uses a simple linear reward model with an explicit confidence bonus. The fact that LinUCB remains competitive across dimensions suggests that much of the useful routing structure in this synthetic MoE setting is captured by a linear reward model.

**Suggested caption for `linucb_minus_online_softmax_vs_d.png`:**

> **Figure X. LinUCB versus online softmax across context dimension.** Negative values indicate dimensions where LinUCB achieves lower cumulative regret. The near-zero gaps show that LinUCB and online softmax are closely matched at (T=10{,}000), with neither method dominating uniformly across dimensions.

The key takeaway should be calibrated carefully:

> LinUCB does not decisively dominate the learned online router, but it matches it closely while using a simpler linear model and explicit uncertainty estimates.

That is more defensible than claiming LinUCB is categorically better.

---

## Result 3: LinUCB Consistently Beats Offline Supervised Softmax

The clearest empirical result is the comparison between LinUCB and the offline supervised softmax router. The offline softmax router is trained with best-arm labels, giving it stronger supervision than LinUCB receives. However, because it is trained offline and then frozen, it cannot adapt during the evaluation stream.

Despite receiving weaker feedback, LinUCB consistently achieves lower regret than the offline softmax baseline in the main joint-trained d-sweep.

**Figure to show:**
Use `linucb_minus_offline_softmax_vs_d.png`.

**Suggested prose:**

The offline supervised softmax router performs well at best-arm classification, but this does not translate into the lowest regret. This distinction is central to the project. Best-arm accuracy measures whether the router selects the exact oracle expert, while regret measures how costly the chosen expert is relative to the oracle. If multiple experts have similar rewards for a context, choosing a non-oracle but near-optimal expert may lead to low regret despite low best-arm accuracy.

This explains why LinUCB can have lower best-arm accuracy but lower regret. LinUCB optimizes reward estimates directly, while the supervised softmax baseline optimizes a classification proxy. In this setting, that proxy is imperfect.

**Suggested caption:**

> **Figure X. LinUCB versus offline supervised softmax.** Negative values indicate lower regret for LinUCB. LinUCB consistently outperforms the frozen supervised router, suggesting that static best-arm classification is not sufficient for minimizing routing regret in this setting.

This is probably your strongest report claim:

> Offline best-arm supervision does not necessarily produce the best routing policy under regret. Routing should be evaluated as a reward-optimization problem, not only as an oracle-arm classification problem.

---

## Result 4: Best-Arm Accuracy and Regret Can Disagree

A recurring pattern in the results is that the offline softmax router often achieves much higher best-arm accuracy than LinUCB, yet suffers higher cumulative regret. This shows that best-arm accuracy is a diagnostic metric, not the primary objective.

In classification terms, the offline router is better at predicting the exact identity of the oracle expert. But in bandit terms, the cost of a mistake depends on the reward gap between the chosen expert and the oracle expert. If the router chooses the second-best expert and its reward is nearly identical to the best expert, the regret is small even though the best-arm prediction is technically wrong.

This matters for MoE routing because expert rewards may be redundant or smooth after joint training. Multiple experts can produce similar predictions for a given input. In that case, exact oracle-arm recovery is less important than choosing an expert with high reward.

**Optional figure:**
This could be discussed without a figure. If you want a supplementary figure later, an oracle-gap histogram would be useful, but it does not appear to be part of the current packet.

**Suggested prose:**

The disagreement between best-arm accuracy and regret is not a bug in the evaluation; it is one of the main findings. Supervised best-arm training treats all classification mistakes equally, but regret weights mistakes by their reward cost. This makes regret the more appropriate metric for routing quality.

---

## Result 5: Alpha Sweep Shows LinUCB Is Not Overly Sensitive in the Useful Range

We also sweep the LinUCB exploration parameter (\alpha). The results show that moderate values of (\alpha) perform similarly, while very large values lead to worse regret. This suggests that excessive exploration is more harmful than mild under-exploration in this setting.

**Figure to show:**
Use `alpha_sweep.png`.

**Suggested prose:**

The alpha sweep shows a relatively flat performance region for moderate exploration values, followed by worse regret at larger alpha. This pattern suggests that the expert reward landscape is learnable enough that aggressive exploration is unnecessary. Once LinUCB has gathered enough evidence to identify high-reward experts, additional exploration mostly adds cost.

This supports selecting (\alpha) by mean regret per dimension in the main d-sweep. It also shows that the LinUCB result is not driven by an arbitrary single alpha choice.

**Suggested caption:**

> **Figure X. LinUCB alpha sweep.** Moderate alpha values perform similarly, while large alpha values increase cumulative regret. This indicates that over-exploration is more costly than under-exploration in the joint-trained reward landscape.

---

## Result 6: Cumulative Regret Remains Approximately Linear at (T=10{,}000)

The regret curves show that none of the online methods clearly reach an asymptotic sublinear regime within (T=10{,}000). Cumulative regret remains approximately linear, although with different slopes across policies.

**Figure to show:**
Use `regret_curves_per_d.png`, probably in the appendix or supplementary results unless you have space.

**Suggested prose:**

Although LinUCB is motivated by sublinear regret guarantees under linear reward realizability, our finite-horizon experiments show approximately linear cumulative regret at (T=10{,}000). This does not necessarily contradict the theory. The synthetic MoE reward functions are only approximately linear, and misspecification introduces a linear regret component. In addition, the evaluated horizon may be too short for asymptotic behavior to dominate.

Thus, the relevant empirical comparison is not whether regret is visibly sublinear within 10,000 steps, but which router achieves the lowest regret slope under the same frozen expert reward process.

**Suggested caption:**

> **Figure X. Cumulative regret curves by dimension.** Regret remains approximately linear over (T=10{,}000), suggesting that finite-horizon misspecification and exploration costs dominate the observed dynamics.

---

# Discussion of Experimental Findings

Taken together, the experiments support a calibrated conclusion. LinUCB is not universally better than every learned router. In the fair online comparison, LinUCB and online softmax are closely matched, with small gaps that vary by dimension. However, LinUCB consistently outperforms the offline supervised softmax router, despite receiving weaker feedback.

This suggests that the main advantage is not simply “linear bandits beat neural routers.” Rather, the advantage comes from aligning the training signal with the evaluation objective. The offline softmax router is trained to classify the best arm, while LinUCB is trained from rewards and evaluated on regret. When expert rewards are smooth or redundant, exact best-arm classification can be a poor proxy for routing quality.

The experiments also clarify the role of joint training. Independent training creates a reward surface that is too sharply cluster-specialized, increasing linear misspecification. Joint training smooths the reward surface and makes linear reward modeling more viable. However, joint training also reduces the sharpness of the routing problem: if many experts are near-equivalent, simple methods can perform surprisingly well. This explains why epsilon-greedy and online softmax remain competitive in some settings.

The central conclusion is therefore:

> In a frozen jointly trained synthetic MoE, LinUCB is a simple online reward-based router that matches a learned online softmax policy and outperforms a frozen supervised best-arm router on regret. This indicates that MoE routing should be evaluated as a reward-optimization problem, not merely as a best-expert classification problem.

---

# Recommended Plot Placement

## Main Text

Use these in the main report:

1. **`epsilon_rmse_comparison.png`**
   Purpose: justify joint training by showing reduced linear misspecification.

2. **`regret_vs_d.png`**
   Purpose: headline result across all policies and dimensions.

3. **`linucb_minus_online_softmax_vs_d.png`**
   Purpose: fair online-method comparison.

4. **`linucb_minus_offline_softmax_vs_d.png`**
   Purpose: strongest claim that LinUCB beats offline supervised routing.

5. **`alpha_sweep.png`**
   Purpose: show alpha choice is principled and not cherry-picked.

## Appendix / Supplementary

Put these in the appendix:

1. **`reward_heatmap_independent.png`** and **`reward_heatmap_joint.png`**
   Purpose: visual explanation of independent vs joint reward structure.

2. **`regret_curves_per_d.png`**
   Purpose: show cumulative regret trajectories, but not necessary for the main story.

3. **`per_seed_regret_vs_d.png`**
   Purpose: demonstrate variance across seeds.

4. **`joint_training_diagnostics.png`**
   Purpose: document that joint MoE training behaved reasonably.

---

# Shorter Version of the Results Narrative

Use this if you need a tighter report version:

> We first compare independent and joint expert training to motivate the main experimental regime. Independent training creates sharply specialized experts and a less linear reward surface, while joint training produces smoother expert rewards and lower linear approximation error. Since LinUCB relies on approximate linear reward structure, we use the independent regime only as a motivational diagnostic and focus the main experiments on jointly trained frozen experts.
>
> In the main d-sweep, LinUCB is competitive with the online softmax router across dimensions. The gap between the two online methods is small and changes sign across d, suggesting that LinUCB does not dominate the learned online router but matches it closely with a simpler linear reward model.
>
> The clearest result is against the offline supervised softmax router. Although this baseline is trained with best-arm labels, LinUCB consistently achieves lower regret. This shows that best-arm classification is not equivalent to regret minimization. A router can have lower oracle-arm accuracy but lower regret if its mistakes are low-cost. This supports evaluating MoE routing as a reward-optimization problem rather than a pure classification problem.
