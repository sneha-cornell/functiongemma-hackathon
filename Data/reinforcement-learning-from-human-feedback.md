# Reinforcement Learning from Human Feedback (RLHF)

## Overview

Reinforcement Learning from Human Feedback (RLHF) is a training paradigm that aligns language models with human preferences by using human-labeled preference data to train a reward model, which then guides policy optimization through reinforcement learning.

## Core Methodology

RLHF typically involves three phases:

1. **Supervised Fine-Tuning (SFT)**: A pre-trained language model is fine-tuned on high-quality demonstration data for the target task (dialogue, summarization, etc.).

2. **Reward Modeling**: Human labelers provide preferences between pairs of model outputs. A reward model is trained to predict human preferences using maximum likelihood on this preference data.

3. **RL Optimization**: The policy (language model) is optimized to maximize the learned reward while staying close to a reference policy (usually the SFT model) using a KL-divergence constraint.

## Key Characteristics

### Complexity
- RLHF is described as "complex and often unstable" in the DPO paper
- Involves training multiple models: the policy model, reward model, and often a value function
- Requires sampling from the language model during fine-tuning
- Needs significant hyperparameter tuning

### Standard Formulation
The RL objective typically maximizes:
```
E[r(x,y)] - β·D_KL[π(y|x) || π_ref(y|x)]
```
where r is the reward function, π is the policy, π_ref is the reference policy, and β controls the KL penalty strength.

## Sources Agreement

**All sources agree that:**
- RLHF is effective for aligning models with human preferences
- The method involves a reward modeling phase followed by policy optimization
- PPO is the most commonly used algorithm for the RL phase
- KL regularization is essential to prevent the policy from deviating too far from the reference

## Sources Disagreement

### On Necessity of Explicit Reward Modeling

**DPO paper argues:**
- Explicit reward modeling in RLHF can be bypassed
- The reward model is "secretly" implicitly defined by the language model itself
- Direct optimization on preferences is simpler and more stable

**PPO and other sources treat:**
- Reward modeling as a fundamental, necessary component
- The reward model as providing crucial signal for policy learning

### On Complexity vs Performance Trade-offs

**DPO position:**
- RLHF's complexity is unnecessary overhead
- Simpler methods can achieve comparable results
- The instability of actor-critic algorithms is problematic

**DeepSeekMath and Goedel-Prover positions:**
- The complexity is manageable and worthwhile
- RLHF (particularly with variants like GRPO) delivers strong improvements
- With proper engineering, stability issues can be addressed

## Practical Considerations

### Implementation Challenges
- Requires careful tuning of the KL penalty coefficient
- Reward model can be difficult to optimize
- Policy updates need careful management to avoid collapse
- High computational cost from sampling and multiple model evaluations

### Recent Developments
The Goedel-Prover paper introduces **Group Relative Policy Optimization (GRPO)**, which:
- Eliminates the need for a separate value function
- Uses group-based normalization for advantages
- Reduces computational overhead while maintaining effectiveness

## Related Concepts
- [Direct Preference Optimization](direct-preference-optimization.md)
- [Proximal Policy Optimization](proximal-policy-optimization.md)
- [Reward Modeling](reward-modeling.md)
- [KL Divergence Regularization](kl-divergence-regularization.md)
- [Group Relative Policy Optimization](group-relative-policy-optimization.md)
