# KL Divergence Regularization

## Overview

KL (Kullback-Leibler) divergence regularization is a critical component of reinforcement learning for language models, used to prevent the policy from deviating too far from a reference policy during optimization. This technique is central to RLHF, PPO, DPO, and related methods.

## Mathematical Definition

The KL divergence between policy π and reference policy π_ref is:

```
D_KL[π(·|x) || π_ref(·|x)] = E_{y~π(·|x)}[log(π(y|x)/π_ref(y|x))]
```

For language models, this is typically computed as:
```
D_KL = E_{y~π}[∑_t log(π(y_t|x,y_{<t})/π_ref(y_t|x,y_{<t}))]
```
summed over all tokens in the sequence.

## Purpose and Motivation

### Why Regularize?

**Without KL regularization:**
- Policy can drift arbitrarily far from reference
- May degenerate to exploiting reward model weaknesses
- Can lose capabilities from pre-training/SFT
- May generate nonsensical outputs that fool the reward model

**With KL regularization:**
- Policy stays in reasonable region of behavior
- Retains fluency and coherence
- Prevents reward hacking
- Maintains useful pre-trained knowledge

### Intuition

The reference policy (typically the SFT model) represents "reasonable" behavior:
- Fluent, coherent language
- Following instruction format
- General helpfulness

KL penalty encourages the RL-trained policy to stay close to this while optimizing reward.

## Use in Different Algorithms

### Reinforcement Learning from Human Feedback (RLHF)

**Standard formulation:**
```
max E_{x~D,y~π}[r(x,y) - β·D_KL[π(·|x) || π_ref(·|x)]]
```

**Implementation:**
- β is a hyperparameter controlling penalty strength
- Typically β ∈ [0.01, 0.5]
- Can be adapted during training
- Added to reward: r_total(x,y) = r(x,y) - β·log(π(y|x)/π_ref(y|x))

### Direct Preference Optimization (DPO)

**Implicit regularization:**

DPO's objective inherently includes KL regularization:
```
L_DPO = -E[log σ(β log(π(y_w|x)/π_ref(y_w|x)) - β log(π(y_l|x)/π_ref(y_l|x)))]
```

The β parameter serves the same role as in RLHF, though the optimization is different.

**Key insight:** The KL constraint in the original RL objective leads to the closed-form optimal policy, which DPO optimizes directly.

### Proximal Policy Optimization (PPO)

**Two approaches to KL control:**

1. **KL penalty in reward:**
   ```
   r_total = r_φ(x,y) - β·log(π(y|x)/π_ref(y|x))
   ```
   Standard approach in RLHF applications

2. **Adaptive KL coefficient:**
   - Monitor D_KL during training
   - If D_KL > target: increase β
   - If D_KL < target: decrease β
   - Automatically maintain desired KL

### Group Relative Policy Optimization (GRPO)

**Direct KL term in loss:**

```
L_GRPO = L_PPO - β·D_KL[π_θ || π_ref]
```

Unlike PPO-style RLHF, GRPO adds KL directly to the loss rather than to the reward, avoiding complications with advantage estimation.

## Choosing the Reference Policy

### Common Choices

**Supervised Fine-Tuned (SFT) Model:**
- Most common choice
- Represents post-instruction-tuning behavior
- Good baseline of helpful, safe responses

**Initial Pre-trained Model:**
- Sometimes used in earlier work
- May be too far from desired behavior
- Less common now

**Previous Policy Checkpoint:**
- In iterative RL, can use policy from previous iteration
- Allows continued divergence over iterations
- Used in some advanced techniques

### Effects of Reference Choice

**Closer reference (e.g., recent checkpoint):**
- Smaller allowed deviation
- More conservative updates
- Slower improvement but more stable

**Farther reference (e.g., SFT model):**
- Larger allowed changes
- Faster potential improvement
- More risk of instability

## Hyperparameter Selection (β)

### Role of β

**Larger β (stronger penalty):**
- Policy stays closer to reference
- More conservative, stable training
- May limit improvement potential
- Less exploration

**Smaller β (weaker penalty):**
- Policy can deviate more
- Potentially larger improvements
- Higher risk of reward hacking
- More exploration

### Typical Values

**From the sources:**
- DPO: β = 0.1 to 0.5
- PPO-based RLHF: β = 0.01 to 0.1
- GRPO: β = 0.04 in Goedel-Prover experiments

**General guidance:**
- Start with moderate values (β ≈ 0.1)
- Tune based on observed KL divergence
- Monitor both reward and KL during training

### Adaptive Methods

**Some implementations adapt β:**

```python
if D_KL > target_KL * 1.5:
    β *= 2
elif D_KL < target_KL / 1.5:
    β /= 2
```

**Advantages:**
- Automatically maintains desired KL
- Reduces need for manual tuning
- Can adapt to different training phases

**From PPO paper:**
- Adaptive KL penalty performs well
- Helps maintain consistent constraint satisfaction

## Measuring KL Divergence

### Online Estimation

**During RL training:**
```
D_KL ≈ (1/N) ∑_i log(π_θ(y_i|x_i)/π_ref(y_i|x_i))
```
where samples y_i ~ π_θ

**Practical considerations:**
- Computed over sampled completions
- Can be noisy with small batches
- Often computed per-token then summed

### Challenges

**Computational cost:**
- Requires forward passes through both π and π_ref
- Can be expensive for large models
- Sometimes approximate or cache reference model outputs

**Variance:**
- Estimate varies across batches
- May need large batches for stable estimates
- Moving averages can help

## Theoretical Justification

### Constrained Optimization Perspective

KL-regularized RL can be viewed as solving:
```
max E[r(x,y)]
subject to: D_KL[π || π_ref] ≤ δ
```

By Lagrangian duality, this is equivalent to:
```
max E[r(x,y)] - β·D_KL[π || π_ref]
```

for some β corresponding to constraint tightness δ.

### From DPO Paper

The paper proves that under certain preference models (Bradley-Terry, Plackett-Luce), the optimal policy for the KL-constrained objective has a closed form:

```
π*(y|x) = (1/Z(x)) π_ref(y|x) exp((1/β)r*(x,y))
```

where Z(x) is a normalization constant and r* is the optimal reward.

This theoretical result underlies DPO's approach.

## Sources Agreement

**All sources agree that:**
- KL regularization is essential for stable RL training
- Prevents reward hacking and mode collapse
- Reference policy should be reasonable baseline (usually SFT model)
- β is an important hyperparameter requiring tuning

## Sources Disagreement

### On Implementation

**PPO/RLHF approaches:**
- Add KL penalty to reward
- May use adaptive β
- Part of standard RL objective

**GRPO approach:**
- Adds KL directly to loss function
- Claims this avoids complications with advantage estimation
- May be more principled

**DPO approach:**
- KL constraint built into closed-form solution
- β parameter plays same role but in different formulation
- No explicit KL computation during training

### On Necessity of Explicit Monitoring

**Some approaches:**
- Carefully monitor KL divergence
- Adjust β if KL grows too large
- Treat as primary safety mechanism

**Other approaches (e.g., DPO):**
- Trust that implicit regularization is sufficient
- Focus on final performance metrics
- Less emphasis on KL monitoring

## Practical Considerations

### When KL Penalty is Critical

**Essential for:**
- Long training runs
- High-capacity models (more prone to overfitting)
- When reward model may be imperfect
- Online RL (where policy updates in loop)

### Monitoring Strategy

**Good practice:**
1. Track mean KL divergence each batch
2. Monitor maximum KL (detect outliers)
3. Plot KL vs. reward over training
4. Check that KL stays bounded
5. Validate quality of high-KL outputs

### Warning Signs

**Problems indicated by:**
- Rapidly increasing KL divergence
- Correlation between high KL and low true quality
- Model generating nonsensical outputs with high reward
- Loss of capabilities from base model

**Solutions:**
- Increase β
- Roll back to earlier checkpoint
- Improve reward model
- Use ensemble of reward models

## Alternatives and Extensions

### Trust Region Methods

**TRPO (Trust Region Policy Optimization):**
- Hard constraint on KL: D_KL ≤ δ
- More principled but more complex
- Less common for language models

### Proximal Penalty

**PPO's clipping:**
- Implicitly limits policy change
- Alternative to direct KL penalty
- Often combined with KL penalty

### Entropy Regularization

**Sometimes added:**
```
L = E[r(x,y)] - β·D_KL[π || π_ref] + α·H[π]
```

where H[π] is entropy, encouraging exploration.

## Related Concepts
- [Reinforcement Learning from Human Feedback](reinforcement-learning-from-human-feedback.md)
- [Direct Preference Optimization](direct-preference-optimization.md)
- [Proximal Policy Optimization](proximal-policy-optimization.md)
- [Group Relative Policy Optimization](group-relative-policy-optimization.md)
- [Reward Modeling](reward-modeling.md)
- [Policy Gradient Methods](policy-gradient-methods.md)
