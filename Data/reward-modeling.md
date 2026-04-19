# Reward Modeling

## Overview

Reward modeling is the process of learning a function that predicts how good a model's output is, typically from human preference data. In RLHF pipelines, the reward model serves as a surrogate for human judgment and guides the RL optimization phase.

## Core Approach

### Preference Data Collection

The standard approach collects preferences through:

1. **Sampling**: Generate multiple outputs (y₁, y₂) for input x from a policy (often the SFT model)
2. **Labeling**: Humans indicate which output they prefer: y_w ≻ y_l | x
3. **Dataset**: Collect dataset D = {(x^(i), y_w^(i), y_l^(i))}ᵢ₌₁ᴺ

### Training Objective

A reward model r_φ(x,y) is trained to maximize the likelihood of human preferences:

```
L_R(r_φ) = -E_{(x,y_w,y_l)~D}[log σ(r_φ(x,y_w) - r_φ(x,y_l))]
```

where σ is the sigmoid function.

This assumes the **Bradley-Terry model**: the probability of preferring y_w over y_l is:

```
p*(y_w ≻ y_l | x) = exp(r*(x,y_w)) / [exp(r*(x,y_w)) + exp(r*(x,y_l))]
                   = σ(r*(x,y_w) - r*(x,y_l))
```

## Implementation

### Architecture

**Common approach:**
- Initialize from the SFT model (same pre-training)
- Add a linear layer on top of the final transformer layer
- Output a single scalar value (the reward)

**Normalization:**
- Often normalize rewards such that E_{y~D}[r_φ(x,y)] = 0 for all x
- Reduces variance and stabilizes RL training

### Training Details

**From the DPO paper:**
- Use the same model architecture as the base LM
- Add a linear projection to produce scalar rewards
- Train with binary cross-entropy loss on preferences
- May require careful hyperparameter tuning

## Use in RLHF Pipeline

After training the reward model:

1. **Freeze** the reward model parameters
2. **Sample** outputs from the policy during RL training
3. **Score** each output using r_φ(x,y)
4. **Optimize** the policy to maximize expected reward with KL penalty:
   ```
   max E_{x~D,y~π}[r_φ(x,y) - β·log(π(y|x)/π_ref(y|x))]
   ```

## Types of Reward Models

### Outcome Reward Models

- Provide a single reward for the complete output
- Assess overall quality
- Simpler to train (only need final judgments)
- Used in most RLHF applications

### Process Reward Models

- Provide rewards at intermediate steps
- More fine-grained feedback
- Require step-level annotations
- Better for complex reasoning tasks

**Goedel-Prover uses process rewards:**
- Verifier feedback at each proof step
- Formal correctness signals
- Enables learning from partial proofs

## Challenges

### Data Quality

**Critical factors:**
- Quality of human preferences
- Consistency of labelers
- Coverage of input distribution
- Sufficient diversity in responses

### Reward Model Limitations

**Known issues:**
- May not generalize beyond training distribution
- Can be exploited through reward hacking
- Variance in predictions can affect policy learning
- Over-optimization can lead to reward model exploitation

### Over-Optimization

**The problem:**
- Policies can find adversarial examples that fool the reward model
- High reward ≠ high quality beyond a point
- KL penalty helps but doesn't completely solve this

**Evidence from literature:**
- Performance often plateaus or degrades with excessive RL training
- Need to carefully monitor actual quality, not just reward

## Sources Agreement

**All sources agree that:**
- Reward models are a key component of traditional RLHF
- Bradley-Terry model is standard for preference modeling
- Quality of reward model significantly impacts final policy quality
- Some form of regularization (KL penalty) is necessary

## Sources Disagreement

### On Necessity of Explicit Reward Models

**Traditional RLHF (PPO paper, standard practice):**
- Explicit reward models are fundamental
- Separating reward learning from policy learning is natural
- Allows for careful reward model validation

**DPO position:**
- Explicit reward modeling is unnecessary
- The optimal policy implicitly defines the reward
- Direct optimization on preferences is cleaner
- Quote: "Your Language Model is Secretly a Reward Model"

**Goedel-Prover position:**
- Uses reward models for process supervision
- But explores efficient variants (GRPO) that rely less on critics
- Suggests reward models are useful but not always in traditional form

### On Reward Model Architecture

**Standard approach:**
- Separate reward model with same architecture as policy
- Trained independently from policy

**Alternative approaches:**
- Share parameters between policy and reward/value functions
- Use ensemble of reward models
- Leverage formal verifiers as reward models (Goedel-Prover)

## Formal Verification as Reward

**Goedel-Prover introduces a unique approach:**

Instead of learning from human preferences:
- Use Lean's **verifier** as reward model
- Provides precise binary feedback: proof correct or incorrect
- No need for human labeling
- Eliminates reward model errors for theorem proving

**Advantages:**
- Perfect accuracy within the formal system
- No generalization issues
- Can provide step-level (process) feedback

**Limitations:**
- Only applicable to formally verifiable tasks
- Doesn't capture human preferences for style, clarity, etc.

## Reward Hacking

### The Problem

Policies can learn to exploit reward model weaknesses:
- Generate outputs that score high but are actually poor quality
- Find edge cases where reward model is wrong
- Maximize proxy metric rather than true objective

### Mitigation Strategies

1. **KL regularization**: Keep policy close to reference
2. **Ensemble methods**: Use multiple reward models
3. **Human-in-the-loop**: Periodically validate with humans
4. **Reward model retraining**: Update reward model during RL
5. **Early stopping**: Don't over-optimize

## Practical Considerations

### When Explicit Reward Models Make Sense

**Good for:**
- Tasks where human feedback is expensive (can reuse reward model)
- Process supervision (step-by-step guidance)
- Iterative training (can update reward model)
- Multiple objectives (train separate reward models)

### When to Consider Alternatives

**Alternatives like DPO are attractive when:**
- Preference data is plentiful
- Simplicity and stability are priorities
- Offline training is sufficient
- Computational resources are limited

### Reward Model Validation

**Important to check:**
- Agreement with human judgments on held-out data
- Generalization to out-of-distribution inputs
- Consistency across similar inputs
- Resistance to adversarial examples

## Related Concepts
- [Reinforcement Learning from Human Feedback](reinforcement-learning-from-human-feedback.md)
- [Direct Preference Optimization](direct-preference-optimization.md)
- [Bradley-Terry Model](bradley-terry-model.md)
- [Preference Learning](preference-learning.md)
- [Process Supervision](process-supervision.md)
- [Outcome Supervision](outcome-supervision.md)
- [KL Divergence Regularization](kl-divergence-regularization.md)
