# Mathematical Reasoning with Language Models

## Overview

Mathematical reasoning is one of the most challenging domains for language models, requiring precise logical thinking, multi-step problem solving, and the ability to manipulate abstract symbols. Recent work has made significant progress through specialized training approaches combining pre-training, supervised fine-tuning, and reinforcement learning.

## Key Benchmarks

### Competition-Level Mathematics

**MATH Dataset (from DPO, DeepSeekMath):**
- 12,500 competition mathematics problems
- Covers algebra, number theory, geometry, etc.
- Requires sophisticated multi-step reasoning
- State-of-art open-source: ~51.7% (DeepSeekMath-7B with RL)

**GSM8K (from all sources):**
- Grade-school math word problems
- 8,500 examples
- Simpler than MATH but requires reasoning
- State-of-art open-source: 88.2% (DeepSeekMath with RL)

### Formal Mathematics Benchmarks

**miniF2F (from Goedel-Prover):**
- 488 formal mathematical statements in Lean
- High-school to early undergraduate level
- IMO, AIME, Putnam competition problems
- Requires formal proof generation

**PutnamBench (from Goedel-Prover):**
- College-level mathematics competition
- 644 problems across multiple domains
- Extremely challenging

## Training Approaches

### Data Collection and Pre-training

**DeepSeekMath approach:**
1. **Collect mathematical web data** from Common Crawl
   - Use fastText classifier to identify math content
   - Iteratively refine classifier with domain-specific data
   - Final corpus: 120B tokens (7x larger than previous math corpora)

2. **Pre-train from code model**
   - Start with DeepSeek-Coder-Base-v1.5 7B
   - Continue pre-training on math corpus
   - Code training improves mathematical reasoning

**Key insight:** The publicly accessible Common Crawl contains valuable mathematical content if properly filtered.

### Supervised Fine-Tuning

**Standard approach (all sources):**
1. Curate high-quality solution demonstrations
2. Format as instruction-following examples
3. Fine-tune pre-trained model with standard supervised loss

**DeepSeekMath SFT data:**
- Chain-of-Thought (CoT) solutions
- Program-of-Thought (PoT) solutions  
- Tool-integrated reasoning
- ~776K training examples

**Goedel-Prover SFT data:**
- Formal proof statements and solutions
- Both vanilla whole-proof generation
- Self-correction training data

### Reinforcement Learning

**All sources demonstrate RL improves mathematical reasoning:**

**DeepSeekMath results:**
- Base + SFT: GSM8K 82.9%, MATH 46.8%
- After RL (GRPO): GSM8K 88.2%, MATH 51.7%
- First open-source model >50% on MATH

**Goedel-Prover results:**
- Significant improvements on formal theorem proving
- Benefits from process supervision via verifier feedback

**DPO results:**
- Achieves >50% on MATH using DPO for RL
- Shows RL-free approach can also work well

## Chain-of-Thought vs Program-of-Thought

### Chain-of-Thought (CoT)

**Characteristics:**
- Natural language reasoning steps
- Explains logic in words
- More interpretable to humans
- Can handle broad reasoning types

**Example:** "Let's solve this step by step. First, we find..."

### Program-of-Thought (PoT)

**Characteristics:**
- Expresses reasoning as code
- Can execute to get numerical answers
- More precise for computation
- Leverages model's code training

**Example:** Generates Python code with math libraries

**DeepSeekMath uses both:**
- Trains on mixture of CoT and PoT solutions
- Tool-integrated reasoning: model decides when to call Python
- Achieves strong results with both approaches

## Tool Use vs Tool-Free Reasoning

### Tool-Integrated Reasoning

**Capabilities:**
- Call external tools (Python interpreter, calculators)
- Verify numerical computations
- More reliable for complex calculations

**DeepSeekMath performance with tools:**
- GSM8K+Python: 86.7%
- MATH+Python: 58.8%

### Tool-Free Reasoning

**Capabilities:**
- Generate complete solutions without external execution
- Tests pure reasoning ability
- More challenging, closer to human solving

**DeepSeekMath performance without tools:**
- GSM8K: 88.2% (after RL)
- MATH: 51.7% (after RL)

**Important finding:** RL significantly improves tool-free performance, suggesting the model learns better reasoning, not just tool use.

## Formal Theorem Proving

### Overview

Formal theorem proving requires:
- Precise logical reasoning
- Generating proofs in formal languages (Lean, Isabelle, Coq)
- Satisfying automatic verification
- Long-horizon planning (proofs can be many steps)

### Approaches

**Whole-Proof Generation (Goedel-Prover):**
- Generate complete proof in one pass
- Uses long chain-of-thought reasoning
- Verifier provides binary feedback

**Proof Search:**
- Iteratively build proof step-by-step
- Search over proof tactics
- More computationally expensive

**Goedel-Prover combines both:**
- Primary approach: whole-proof generation with self-correction
- Can iteratively refine proofs using verifier feedback

### Self-Correction

**Key innovation in Goedel-Prover:**

1. Generate initial proof attempt
2. If verification fails, provide error feedback to model
3. Model attempts to correct the proof
4. Repeat up to N times

**Results:**
- 2 rounds of self-correction significantly improves pass rate
- miniF2F: 84.6% → 90.4% (32B model with self-correction)
- Shows models can learn from their mistakes with precise feedback

### Scaffolded Data Synthesis

**Goedel-Prover introduces two strategies:**

1. **Formal-based scaffolding:**
   - Use Lean tactics (extract_goal) to extract sub-goals
   - Create easier sub-problems from failed proofs
   - Train on graduated difficulty

2. **Informal-based scaffolding:**
   - Generate simpler/sub-problems in natural language
   - Have model solve and formalize them
   - Filter for correctness and difficulty

**Purpose:** Create training data at appropriate difficulty levels to help model learn incrementally complex reasoning.

## Sources Agreement

**All sources agree that:**
- Mathematical reasoning is very challenging for LLMs
- Multi-stage training (pre-training → SFT → RL) is effective
- Code training benefits mathematical reasoning
- RL/post-training significantly improves performance
- Larger models generally perform better
- Scale of pre-training data matters

## Sources Disagreement

### On the Role of RL

**DeepSeekMath and Goedel-Prover emphasize:**
- RL (specifically GRPO variants) provides substantial gains
- Process supervision is valuable
- Iterative training with RL enables continued improvement

**DPO paper demonstrates:**
- Strong results possible with simpler offline preference learning
- Explicit RL may not be necessary
- Direct optimization can be equally effective

### On Data Sources

**DeepSeekMath finds:**
- Common Crawl math content is valuable
- Properly filtered web data rivals curated datasets
- Multilingual data helps (includes Chinese math content)

**Traditional approaches (implicit in comparisons):**
- Focus on curated sources like arXiv, textbooks
- May under-utilize available web data

### On Formalization

**Goedel-Prover focus:**
- Formal verification provides perfect training signal
- Formalization is the future of mathematical reasoning
- Verifier-guided training is ideal

**DeepSeekMath focus:**
- Natural language mathematical reasoning
- Informal proofs and problem solving
- Tool integration for verification

**Note:** These aren't contradictory—just different domains with different requirements.

## Key Insights from Sources

### Code Training Improves Math Reasoning

**DeepSeekMath insight:**
- Starting from code model (DeepSeek-Coder-Base) better than general LLM
- Code and math share structured reasoning
- Best of both worlds: pre-train on code, then math

**Supporting evidence:**
- Ablation studies show benefit
- PoT (program-of-thought) leverages code abilities
- Consistent with broader literature

### Scale of Math Pre-training Matters

**DeepSeekMath demonstrates:**
- 120B token math corpus significantly outperforms smaller corpora
- Comparable to Minerva 540B (closed-source) despite 77x fewer parameters
- Quality and quantity of domain-specific pre-training is crucial

### Process Supervision Enables Complex Reasoning

**Goedel-Prover shows:**
- Step-by-step verifier feedback (process rewards)
- Enables learning from partial progress
- Particularly important for long-horizon tasks like theorem proving

### Self-Correction is Learnable

**Goedel-Prover demonstrates:**
- Models can learn to fix their own errors
- Requires precise feedback (verifier messages)
- Iterative refinement substantially improves success rates

## Practical Considerations

### For Natural Language Math

**Effective strategies:**
1. Pre-train or continue-train on large math corpus
2. Start from code-trained model if possible
3. Create diverse SFT data with CoT and PoT
4. Apply RL/preference learning for further gains
5. Enable tool use for complex calculations

### For Formal Theorem Proving

**Effective strategies:**
1. Train on formal proof corpora
2. Use verifier feedback for RL
3. Implement self-correction loops
4. Generate synthetic data at appropriate difficulty
5. Combine whole-proof generation with search when needed

### Model Size Considerations

**Findings across sources:**
- 7B models can achieve strong performance with good data/training
- 32B+ models reach higher absolute performance
- Efficiency techniques (model averaging, etc.) help smaller models

## Related Concepts
- [Reinforcement Learning from Human Feedback](reinforcement-learning-from-human-feedback.md)
- [Chain-of-Thought Reasoning](chain-of-thought-reasoning.md)
- [Tool-Integrated Reasoning](tool-integrated-reasoning.md)
- [Formal Theorem Proving](formal-theorem-proving.md)
- [Process Supervision](process-supervision.md)
- [Self-Correction](self-correction.md)
- [Group Relative Policy Optimization](group-relative-policy-optimization.md)
