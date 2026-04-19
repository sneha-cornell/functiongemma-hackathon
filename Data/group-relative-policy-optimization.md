# Group Relative Policy Optimization (GRPO)

## Overview

Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm introduced in the Goedel-Prover-V2 paper as an efficient and effective alternative to PPO for training language models from preferences. GRPO eliminates the need for a value function while maintaining the benefits of policy optimization.

## Core Innovation

GRPO's key innovation is using **group-based relative rewards** instead of traditional value function baselines:

1. For each question, sample a **group** of outputs (e.g., 64 completions)
2. Score all outputs in the group using the reward model
3. Normalize rewards **within each group** (subtract mean, divide by std dev)
4. Use these relative rewards as advantages for policy optimization

## Motivation

### Problems with Traditional PPO

**Traditional PPO requires:**
- A separate value function V(s) for variance reduction
- Training this critic alongside the policy
- Additional model parameters and computation

**The value function is problematic because:**
- Adds computational overhead
- Requires careful tuning
- In language models, typically only the last token gets a reward score
- The value function may not accurately estimate returns at each token

### GRPO's Solution

**By using group normalization:**
- No separate value function needed
- Advantages computed directly from group statistics
- Aligns with how reward models work (comparing outputs for same question)
- Reduces training resources significantly

## Mathematical Formulation

### Objective

GRPO optimizes:
```
J_GRPO(θ) = E[q ~ P(Q), {o_i}^G_i=1 ~ π_old(·|q)] [
    (1/G) ∑_i=1^G (1/|o_i|) ∑_t=1^|o_i| [
        min(r_i(θ) · A_i,t, clip(r_i(θ), 1-ε, 1+ε) · A_i,t)
    ] - β·D_KL[π_θ || π_ref]
]
```

where:
- G is the group size
- {o_i} is a group of outputs for question q
- r_i(θ) is the probability ratio
- A_i,t is the advantage at token t

### Advantage Calculation

For each group, advantages are computed as:

**Outcome Supervision (reward at end):**
```
A_i,t = (r_i - mean(r_group)) / std(r_group)
```
where r_i is the reward for output i, and statistics are over the group.

**Process Supervision (reward at each step):**
```
A_i,t = ∑_j=index(t)^K (r_j^{index(i)} - mean(R)) / std(R)
```
where rewards are given at each reasoning step.

### KL Divergence Handling

Unlike PPO, GRPO:
- Adds KL divergence directly to the loss
- Avoids computing KL penalty in the reward
- Regularizes by comparing trained policy to reference policy

## Implementation Details

### From Goedel-Prover-V2

**Training setup:**
- Multi-task: 90% whole-proof generation, 10% self-correction
- Group size: 64 outputs per question
- Single epoch per update
- Learning rate: 1e-6 for policy model
- Max length: 1024 tokens
- Batch size: 1024 training examples

### Key Modifications from Vanilla GRPO

**Goedel-Prover introduces:**
1. **Dynamic sampling strategy**: Only includes problems with pass rates in [0, 0.75] during optimization
2. **No group normalization** (as suggested by Dr.GRPO)
3. **Dynamic sampling from DAPO**: Excludes easy problems and very hard problems
4. **Incorporating past data**: Uses replay mechanism with 10% historical data

## Performance

### Advantages Over PPO

**Computational efficiency:**
- No value function to train
- Fewer parameters
- Reduced memory usage
- Faster training

**Effectiveness:**
- Comparable or better performance than PPO
- More stable training dynamics
- Better handles varying output lengths

### Results from Goedel-Prover-V2

**Mathematical reasoning improvements:**
- DeepSeekMath-Instruct: GSM8K: 82.9% → 88.2%, MATH: 46.8% → 51.7%
- Strong gains on out-of-domain tasks (CMATH: 84.6%, up from 73.2%)
- Demonstrates that RL enhances both in-domain and out-of-domain performance

**Formal theorem proving:**
- Substantial improvements over instruction-tuned baseline
- Shows RL is effective for complex reasoning tasks beyond natural language

## Outcome vs Process Supervision

GRPO supports two reward schemes:

### Outcome Supervision (OS)
- Reward given only at the end of each output
- Single reward per completion
- Simpler to implement
- Works well for problems with verifiable final answers

### Process Supervision (PS)
- Rewards given at intermediate reasoning steps
- Requires step-by-step annotations or verification
- Can provide more granular feedback
- Better for complex multi-step reasoning

**In Goedel-Prover:** Process rewards use formal verification feedback at each proof step, providing precise correctness signals for theorem proving.

## Comparison with Other Methods

### vs PPO
- **GRPO**: No value function, group-based advantages, more efficient
- **PPO**: Requires critic, token-level value estimates, more complex

### vs DPO
- **GRPO**: Still uses RL framework, samples from policy online, can iterate
- **DPO**: Offline, no sampling needed, simpler but less flexible

### vs Rejection Sampling Fine-Tuning (RFT)
- **GRPO**: Uses reward-weighted updates, explores broadly
- **RFT**: Only trains on correct samples, may be less sample-efficient

## Iterative Training

Goedel-Prover demonstrates **iterative RL** with GRPO:

1. Train initial reward model and policy with GRPO
2. Generate new training data using current policy
3. Update reward model with new data (replay mechanism)
4. Train policy again with updated reward model
5. Repeat

**Results show:**
- Iteration 1 provides significant gains
- Further iterations continue to improve performance
- Combining with model averaging enhances results

## Sources Agreement

**Goedel-Prover paper shows:**
- GRPO is effective for mathematical reasoning and theorem proving
- Group normalization provides sufficient variance reduction
- The method is more resource-efficient than traditional PPO

## Sources Disagreement

### On Necessity of RL Framework

**GRPO maintains:**
- RL framework has value for iterative learning
- Online sampling and policy updates enable exploration
- Process rewards benefit from RL formulation

**DPO argues:**
- Direct optimization is sufficient
- RL complexity can be avoided entirely
- Outcome-supervised preference learning works well

### On Value Function Necessity

**GRPO demonstrates:**
- Value functions are not essential for variance reduction
- Group-based baselines work effectively

**Traditional PPO practitioners:**
- Value functions are important for sample efficiency
- Critics help stabilize training

## Practical Considerations

### When to Use GRPO

**GRPO is particularly suitable when:**
- Training resources are limited (compared to full PPO)
- You have natural groupings of outputs (e.g., multiple attempts per problem)
- Process-level feedback is available
- Iterative improvement is desired

### Hyperparameter Choices

**Key hyperparameters:**
- Group size G (typically 64)
- Clipping parameter ε (typically 0.04 in Goedel-Prover)
- KL coefficient β
- Dynamic sampling range (e.g., [0, 0.75] pass rate)

## Related Concepts
- [Reinforcement Learning from Human Feedback](reinforcement-learning-from-human-feedback.md)
- [Proximal Policy Optimization](proximal-policy-optimization.md)
- [Direct Preference Optimization](direct-preference-optimization.md)
- [Reward Modeling](reward-modeling.md)
- [Process Supervision](process-supervision.md)
- [Outcome Supervision](outcome-supervision.md)
