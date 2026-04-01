# dhant Paper Map

This document maps each trainer in `dhant` to foundational research papers.

The goal is educational clarity: each entry includes a core paper, why it matters, and how it connects to the corresponding trainer.

## Trainer-to-Paper Mapping

## 1. SFTTrainer (Supervised Fine-Tuning)

### Paper A
- Title: Training language models to follow instructions with human feedback
- ArXiv: https://arxiv.org/abs/2203.02155
- Authors: Long Ouyang et al. (OpenAI)
- Description: Introduces the now-standard post-training pipeline with supervised instruction tuning followed by preference optimization and RL.
- Why this paper for `SFTTrainer`: It clearly defines the SFT stage as the first alignment step, where a pretrained model is fine-tuned on high-quality demonstrations.

### Paper B
- Title: Scaling Instruction-Finetuned Language Models
- ArXiv: https://arxiv.org/abs/2210.11416
- Authors: Hyung Won Chung et al. (Google Research)
- Description: Studies instruction fine-tuning at scale and demonstrates strong zero-shot and few-shot generalization gains.
- Why this paper for `SFTTrainer`: It gives practical evidence that instruction data quality and diversity strongly affect downstream behavior.

## 2. DPOTrainer (Direct Preference Optimization)

### Core Paper
- Title: Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- ArXiv: https://arxiv.org/abs/2305.18290
- Authors: Rafael Rafailov et al.
- Description: Proposes DPO, which optimizes preference data directly without explicitly training a separate reward model and running PPO.
- Why this paper for `DPOTrainer`: `DPOTrainer` directly implements this formulation, making it one of the cleanest ways to teach preference-based alignment.

## 3. GRPOTrainer (Group Relative Policy Optimization)

### Core Paper
- Title: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- ArXiv: https://arxiv.org/abs/2402.03300
- Authors: Zhihong Shao et al. (DeepSeek-AI)
- Description: Introduces GRPO in a practical large-scale reasoning setup; uses grouped relative comparisons to improve policy learning efficiency.
- Why this paper for `GRPOTrainer`: It is the primary public reference for GRPO and explains why relative, group-based updates can reduce memory and variance challenges seen in PPO-style training.

## 4. RewardTrainer (Reward Model Training)

### Paper A
- Title: Fine-Tuning Language Models from Human Preferences
- ArXiv: https://arxiv.org/abs/1909.08593
- Authors: Daniel M. Ziegler et al. (OpenAI)
- Description: Classic work showing how to learn reward signals from pairwise human preferences.
- Why this paper for `RewardTrainer`: It is foundational for training a reward model before policy optimization.

### Paper B
- Title: Learning to summarize from human feedback
- ArXiv: https://arxiv.org/abs/2009.01325
- Authors: Nisan Stiennon et al. (OpenAI)
- Description: Demonstrates reward modeling and RLHF for summarization, including reward model training and evaluation methodology.
- Why this paper for `RewardTrainer`: It provides concrete reward model design, data collection, and validation patterns that transfer well to modern pipelines.

## Reward Functions and Reward-Model Adjacent Reading

These papers are useful when extending `dhant` reward functions such as accuracy or reasoning-based rewards.

### Process Reward and Reasoning Supervision
- Title: Let’s Verify Step by Step
- ArXiv: https://arxiv.org/abs/2305.20050
- Authors: Hunter Lightman et al.
- Description: Shows that process supervision (rewarding intermediate reasoning steps) can significantly improve math reasoning.
- Relevance to `reasoning_accuracy_reward`: Motivates moving from only final-answer rewards toward richer reasoning-aware rewards.

### AI Feedback and Rule-Guided Reward Signals
- Title: Constitutional AI: Harmlessness from AI Feedback
- ArXiv: https://arxiv.org/abs/2212.08073
- Authors: Yuntao Bai et al. (Anthropic)
- Description: Uses AI-generated critiques/preferences with constitutional principles for scalable alignment.
- Relevance to `format_reward` and policy shaping: Supports rule-based and structure-aware reward ideas when human labels are expensive.

## Suggested Reading Order for Learners

1. 2203.02155 for the full post-training pipeline context.
2. 2210.11416 for instruction tuning scale behavior.
3. 2305.18290 for DPO theory and implementation intuition.
4. 2402.03300 for GRPO and reasoning-oriented RL training.
5. 1909.08593 and 2009.01325 for reward-model foundations.
6. 2305.20050 for process rewards and reasoning quality.

## Notes

- `dhant` is an educational template, not a full reproduction of every paper.
- Use this map as a bridge between implementation files and the primary literature.
