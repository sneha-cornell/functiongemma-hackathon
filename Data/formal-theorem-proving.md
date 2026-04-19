# Formal Theorem Proving

## Overview

Formal theorem proving involves constructing machine-verifiable proofs of mathematical statements in formal proof languages like Lean, Isabelle, or Coq. Unlike informal mathematical reasoning, formal proofs must satisfy rigid syntactic and logical requirements that can be automatically verified.

## Key Challenges

### Verification and Precision

**Requirements:**
- Proofs must be syntactically correct in the formal language
- Every logical step must be justified
- Must satisfy the automated proof checker
- No ambiguity or informal reasoning

**Contrast with informal math:**
- Informal proofs can skip "obvious" steps
- Natural language allows imprecision
- Human judgment determines correctness

### Long-Horizon Reasoning

**Characteristics:**
- Proofs can be very long (many steps/tactics)
- Requires planning and sub-goal management
- Each error can invalidate the entire proof
- Must maintain consistency throughout

### Search Space Complexity

**The challenge:**
- Many possible tactics at each step
- Exponential branching in proof search
- Most paths lead to dead ends
- Difficult to evaluate partial progress

## Approaches to Automated Theorem Proving

### Proof Search Methods

**Traditional approach:**
- Iteratively select proof tactics
- Search through proof tree
- Use heuristics or learned models to guide search
- Examples: Monte Carlo Tree Search, best-first search

**Characteristics:**
- More systematic exploration
- Can handle complex proofs
- Computationally expensive
- Requires many proof attempts

### Whole-Proof Generation

**Modern LLM approach (Goedel-Prover):**
- Generate entire proof in one pass
- Uses long chain-of-thought reasoning
- Single attempt or few refinement iterations
- Much more computationally efficient

**Advantages:**
- Faster inference (fewer verification calls)
- More sample efficient
- Simpler infrastructure

**Challenges:**
- Requires strong long-horizon reasoning
- Must get everything right at once
- Need powerful base models

## Goedel-Prover Methodology

### Verifier-Guided Self-Correction

**Core innovation:**

1. **Initial Generation**: Model generates complete proof
2. **Verification**: Lean verifier checks the proof
3. **Feedback**: If incorrect, verifier error messages provided to model
4. **Correction**: Model attempts to fix the proof
5. **Iterate**: Repeat up to N rounds

**Key insight:** Compiler/verifier feedback provides precise signal about what's wrong, enabling targeted corrections.

**Results:**
- 2 rounds substantially improve success
- miniF2F: 84.6% → 90.4% (32B with self-correction)
- Shows models can learn from specific error feedback

### Training Data Curation

**Components:**

1. **Formalizer Training**: 
   - Translate informal statements to formal Lean
   - Enable working with naturally stated problems

2. **Proof Generation Training**:
   - Formal statements with solutions
   - Mix of successful and corrected proofs

3. **Self-Correction Data**:
   - Failed proof attempts with verifier feedback
   - Corrected versions
   - Teaches error recovery

### Scaffolded Data Synthesis

**Problem:** Need training data at appropriate difficulty levels

**Goedel-Prover's solutions:**

#### Formal-Based Scaffolding

1. Use Lean's `extract_goal` tactic to extract sub-goals from proofs
2. Create standalone problems from these sub-goals
3. Train model on graduated difficulty
4. Invalid sub-goals also included with negation (teaches what's false)

#### Informal-Based Scaffolding

1. Prompt LLM to generate simpler variants of hard problems
2. Filter for correctness (must be solvable)
3. Filter for difficulty (judge simplicity)
4. Train on these synthetic easier problems

**Purpose:** Help model learn incrementally complex reasoning.

### Training Pipeline

**Goedel-Prover's multi-stage approach:**

1. **Supervised Fine-Tuning (SFT)**:
   - Initial training on formal proof pairs
   - Multiple SFT iterations with growing datasets

2. **Expert Iteration**:
   - Generate proofs for training problems
   - Keep correct proofs
   - Augment training set

3. **Reinforcement Learning (GRPO)**:
   - Use verifier feedback as rewards (process supervision)
   - Optimize for proof correctness
   - Iterate with new data from policy

4. **Model Averaging**:
   - Average parameters from multiple checkpoints
   - Improves output diversity
   - Reduces over-fitting

## Results and Performance

### Goedel-Prover-V2 Performance

**miniF2F (pass@32):**
- 8B model: 84.6% → 89.3% (with self-correction)
- 32B model: 88.1% → 90.4% (with self-correction)

**Comparison to prior work:**
- Substantial improvement over previous SOTA
- 32B model outperforms 671B DeepSeek-Prover-V2
- More efficient: 80x fewer parameters for comparable performance

**PutnamBench:**
- 86 problems solved
- Best among open-source models
- 39 more problems than previous SOTA DeepSeek-Prover-V2

### Scaling Behavior

**Key finding from Goedel-Prover:**
- Performance improves across inference budgets (pass@N)
- Gains from verifier-guided self-correction hold across budgets
- Smaller models (8B) competitive with much larger models

**Efficiency:**
- 1-2 percentage point improvement for 1-2 self-correction rounds
- Diminishing returns beyond 2 rounds
- Sweet spot: 2 rounds of correction

## Verifier as Reward Model

### Advantages

**Perfect precision within the formal system:**
- No uncertainty about correctness
- Binary feedback: proof works or doesn't
- No need for human labeling

**Process-level feedback:**
- Error messages indicate where/why proof fails
- Can identify specific incorrect steps
- Enables targeted learning

**Scalability:**
- Verification is fast (milliseconds to seconds)
- Can generate unlimited training signal
- No bottleneck on human feedback

### Comparison to Learned Reward Models

**Formal verifier:**
- ✓ Perfect accuracy for correctness
- ✓ Free to use (no human annotation)
- ✓ Precise error feedback
- ✗ Only applicable to formal domains
- ✗ Doesn't capture style/elegance preferences

**Learned reward model:**
- ✓ Applicable to any domain
- ✓ Can capture subjective preferences
- ✗ May make mistakes
- ✗ Requires human-labeled training data
- ✗ May not generalize well

## Sources Agreement

**Goedel-Prover demonstrates:**
- Whole-proof generation with self-correction is effective
- Verifier feedback enables strong RL training
- Scaffolded data synthesis helps learning
- Model averaging improves diversity and performance

## Contrasts with Informal Mathematical Reasoning

### Formal Theorem Proving (Goedel-Prover)

**Characteristics:**
- Precise formal language (Lean)
- Automatic verification provides perfect feedback
- Binary outcome: proof correct or not
- Process rewards from verifier

**Advantages:**
- No ambiguity in correctness
- Training signal is free and accurate
- Can iterate indefinitely with verification

**Limitations:**
- Only applies to formalized problems
- Formalization itself is challenging
- Limited corpus of formalized mathematics

### Informal Mathematical Reasoning (DeepSeekMath)

**Characteristics:**
- Natural language or code
- Human evaluation or answer-checking
- More flexible reasoning
- Broader problem coverage

**Advantages:**
- Applicable to any math problem
- Easier data collection
- More natural problem statements

**Limitations:**
- Harder to verify correctness (except numerical answers)
- Process supervision is expensive
- Ambiguity in what constitutes good reasoning

## Practical Considerations

### When to Use Formal Methods

**Formal theorem proving is suitable when:**
- Proofs must be verified with certainty
- Working within an established formal library
- Have or can create formalizations
- Need absolutely correct results

### Implementation Considerations

**Key requirements:**
- Formal proof language environment (Lean, Isabelle, etc.)
- Fast verifier for quick feedback
- Good base model (trained on code helps)
- Infrastructure for iterative verification

### Training Strategy

**Effective approach (from Goedel-Prover):**
1. Start with quality formal proof corpus
2. Use expert iteration to augment data
3. Generate scaffolded synthetic data
4. Apply RL with verifier rewards
5. Implement self-correction at inference time
6. Use model averaging

## Future Directions

**Potential improvements:**
- Better formalizers (informal → formal translation)
- More efficient proof search combined with generation
- Transfer learning between formal systems
- Larger formal proof corpora
- Better synthetic data generation

**Open questions:**
- How to handle truly novel proofs?
- Can models discover new proof techniques?
- How to balance proof length vs. readability?
- Optimal number of self-correction rounds?

## Related Concepts
- [Mathematical Reasoning with Language Models](mathematical-reasoning-with-language-models.md)
- [Process Supervision](process-supervision.md)
- [Self-Correction](self-correction.md)
- [Verifier-Guided Learning](verifier-guided-learning.md)
- [Reinforcement Learning from Human Feedback](reinforcement-learning-from-human-feedback.md)
- [Group Relative Policy Optimization](group-relative-policy-optimization.md)
- [Scaffolded Data Synthesis](scaffolded-data-synthesis.md)
