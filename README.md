# Lightweight RLHF Pipeline for Aligning LLMs

> A full replication of Anthropic's three-stage RLHF training pipeline at academic scale, plus a novel ablation study on reward model capacity.
>
> **Authors:** Kumi Acheampong · Selase Doku  
> *CIS-585 — University of Michigan*

---

## Overview

Large language models trained purely on next-token prediction don't inherently behave helpfully or safely. This project implements the complete **Reinforcement Learning from Human Feedback (RLHF)** pipeline — as described by Ouyang et al. (2022) and Bai et al. (2022) — using consumer-grade hardware and open datasets.

We go beyond replication with a **novel ablation**: testing whether reward model capacity (117M vs. 14M parameters) bottlenecks downstream alignment quality in PPO. It does — by approximately 6 percentage points in win rate.

**Key results:**

| Model | Win Rate vs. SFT | RM Accuracy | Perplexity |
|---|---|---|---|
| SFT Baseline | 50% | — | 24.1 |
| RLHF (Full RM, 117M) | **67%** | 78.2% | **22.8** |
| RLHF (Small RM, 14M) | 61% | 71.4% | 23.3 |

---

## Pipeline

```
Pre-trained LM (GPT-2)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Stage 1: Supervised Fine-Tuning (SFT)          │
│  Loss: Cross-entropy on chosen responses        │
│  Data: Anthropic HH-RLHF helpful split (20K)   │
│  Method: LoRA (r=16, α=32) — 99% params frozen │
└────────────────────┬────────────────────────────┘
                     │  π_SFT
                     ▼
┌─────────────────────────────────────────────────┐
│  Stage 2: Reward Model Training                 │
│  Loss: Bradley-Terry (pairwise preference)      │
│  Data: HH-RLHF pairwise comparisons (20K pairs)│
│  Ablation: full RM (117M) vs. small RM (14M)   │
└────────────────────┬────────────────────────────┘
                     │  r_φ
                     ▼
┌─────────────────────────────────────────────────┐
│  Stage 3: PPO Fine-Tuning                       │
│  Objective: max E[r_φ(x,y)] − β·KL(π_θ‖π_SFT) │
│  Library: TRL (HuggingFace)                     │
│  KL control: adaptive β controller              │
└────────────────────┬────────────────────────────┘
                     │  π_RLHF
                     ▼
┌─────────────────────────────────────────────────┐
│  Evaluation                                     │
│  Win rate · RM accuracy · Perplexity · KL       │
└─────────────────────────────────────────────────┘
```

---

## Project Structure

```
rlhf_pipeline/
│
├── 01_supervised_finetuning.ipynb   # Stage 1: SFT with LoRA on Anthropic HH-RLHF
├── 02_reward_model.ipynb            # Stage 2: Bradley-Terry RM + undersized ablation
├── 03_ppo_finetuning.ipynb          # Stage 3: PPO with KL penalty via TRL
├── 04_evaluation.ipynb              # Win rate, RM accuracy, perplexity, KL analysis
│
├── rlhf_app.py                      # Gradio interactive demo (all pipeline stages)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install transformers>=4.40.0 datasets>=2.19.0 trl>=0.8.6 \
            accelerate>=0.29.0 peft>=0.10.0 bitsandbytes>=0.43.0 \
            torch>=2.2.0 wandb einops sentencepiece gradio
```

### 2. Run the pipeline (in order)

```bash
jupyter notebook 01_supervised_finetuning.ipynb
jupyter notebook 02_reward_model.ipynb
jupyter notebook 03_ppo_finetuning.ipynb
jupyter notebook 04_evaluation.ipynb
```

Checkpoints are saved to `./checkpoints/` between stages. Each notebook loads from the previous stage's output.

### 3. Interactive demo

```bash
python rlhf_app.py
```

Opens a Gradio UI where you can compare SFT vs. RLHF outputs in real time.

---

## Stage Details

### Stage 1 — Supervised Fine-Tuning

Fine-tunes GPT-2 on `(prompt, chosen_response)` pairs from the [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) helpful split using standard cross-entropy loss. LoRA is applied to the attention projections (`c_attn`, `c_proj`) to keep memory requirements low.

**Key config:**

```python
MODEL_NAME     = "gpt2"
LORA_R         = 16
LORA_ALPHA     = 32
TRAIN_SAMPLES  = 20_000
LEARNING_RATE  = 2e-5
NUM_EPOCHS     = 3
```

The resulting checkpoint (`./checkpoints/sft`) becomes π_SFT — the reference policy used throughout PPO training to constrain the KL penalty.

---

### Stage 2 — Reward Model

Trains a scalar reward model `r_φ(x, y)` using the Bradley-Terry probabilistic model:

$$\mathcal{L}_{RM} = -\mathbb{E}\left[\log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

A linear reward head is added on top of the SFT backbone, trained on pairwise preference data from HH-RLHF.

**Novel ablation:** We train two reward models — a full-size RM (117M params, `HIDDEN_DIM=768`) and an undersized RM (14M params, `HIDDEN_DIM=128`) — then compare their effect on PPO alignment quality downstream. This experiment was not reported in Ouyang et al. (2022).

**Result:** The smaller RM achieves only 71.4% preference accuracy vs. 78.2% for the full RM, and leads to a 6pp drop in RLHF win rate — confirming that **reward model capacity is a meaningful bottleneck**.

---

### Stage 3 — PPO Fine-Tuning

Optimizes the policy using proximal policy optimization with a KL divergence penalty:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta}\left[r_\phi(x, y)\right] - \beta \cdot \mathbb{KL}\left[\pi_\theta \,\|\, \pi_\text{SFT}\right]$$

Implemented via [TRL](https://github.com/huggingface/trl). An adaptive β controller adjusts the KL penalty weight throughout training to keep the policy from reward-hacking while still improving over the SFT baseline.

---

### Evaluation

`04_evaluation.ipynb` computes:

- **Win rate** — RLHF vs. SFT, scored by the reward model on held-out prompts
- **RM accuracy** — preference classification accuracy on a held-out validation set
- **Perplexity** — language modeling quality on WikiText-103
- **KL divergence** — tracked throughout PPO training to assess policy drift
- **Ablation comparison** — full RM vs. undersized RM across all metrics

---

## Hardware & Compute

Designed to run on a single GPU with ≥8 GB VRAM (tested on RTX 3080/4090).

| Stage | Approx. Time (RTX 3080) |
|---|---|
| SFT (3 epochs, 20K samples) | ~2 hours |
| Reward Model | ~1.5 hours |
| PPO Fine-Tuning | ~2.5 hours |
| Evaluation | ~30 minutes |

bf16 mixed precision is used automatically on Ampere GPUs (RTX 30xx, A100+). fp16 fallback otherwise.

---

## References

- Ouyang et al. (2022). *Training language models to follow instructions with human feedback.* NeurIPS. [[paper]](https://arxiv.org/abs/2203.02155)
- Bai et al. (2022). *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.* Anthropic. [[paper]](https://arxiv.org/abs/2204.05862)
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* [[paper]](https://arxiv.org/abs/1707.06347)
- von Werra et al. (2022). *TRL: Transformer Reinforcement Learning.* [[repo]](https://github.com/huggingface/trl)
- Anthropic HH-RLHF dataset. [[HuggingFace]](https://huggingface.co/datasets/Anthropic/hh-rlhf)
