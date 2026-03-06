# Lightweight RLHF Pipeline for Aligning LLMs


Replicates the Anthropic RLHF three-stage pipeline at academic scale, 
plus a novel ablation on reward model capacity.

---

## Project Structure

```
rlhf_pipeline/
│
├── 01_supervised_finetuning.ipynb   # Stage 1: SFT with LoRA on Anthropic HH-RLHF
├── 02_reward_model.ipynb            # Stage 2: Bradley-Terry RM + undersized ablation
├── 03_ppo_finetuning.ipynb          # Stage 3: PPO with KL penalty (via TRL)
├── 04_evaluation.ipynb              # Win rate, RM accuracy, perplexity, KL analysis
│
├── rlhf_app.py                      # Gradio interactive UI (all stages)
├── requirements.txt
└── README.md
```

---

## Pipeline Overview

```
Pre-trained LM (GPT-2 / LLaMA)
        │
        ▼
┌─────────────────────┐
│  Stage 1: SFT       │  Fine-tune on (prompt, response) pairs
│  Loss: Cross-entropy│  Dataset: Anthropic HH-RLHF (helpful split)
└─────────┬───────────┘
          │  π_SFT
          ▼
┌─────────────────────┐
│  Stage 2: RM        │  Train scalar reward model r_φ(x,y)
│  Loss: Bradley-Terry│  Dataset: HH-RLHF pairwise prefs (~170K)
└─────────┬───────────┘
          │  r_φ
          ▼
┌─────────────────────┐
│  Stage 3: PPO       │  max E[r_φ(x,y)] - β·KL(π_θ || π_SFT)
│  Algorithm: PPO+KL  │  Adaptive KL controller
└─────────┬───────────┘
          │  π_RLHF
          ▼
┌─────────────────────┐
│  Evaluation         │  Win rate, RM acc, perplexity, KL analysis
└─────────────────────┘
```

