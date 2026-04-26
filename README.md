# Lightweight RLHF Pipeline for Aligning LLMs

**CIS-585: Advanced AI | University of Michigan-Dearborn**  
**Authors:** Kumi Acheampong · Selase Doku

Replicates the InstructGPT / Anthropic RLHF three-stage pipeline at academic scale, 
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

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run notebooks in order
jupyter notebook 01_supervised_finetuning.ipynb
jupyter notebook 02_reward_model.ipynb
jupyter notebook 03_ppo_finetuning.ipynb
jupyter notebook 04_evaluation.ipynb

# 3. Launch interactive Gradio UI
python rlhf_app.py
# → http://localhost:7860
```

---

## Datasets

| Stage | Dataset | Size |
|-------|---------|------|
| SFT | Anthropic HH-RLHF (helpful) | ~20K pairs |
| RM | Anthropic HH-RLHF (pairwise) | ~170K pairs |
| RL | Anthropic HH-RLHF (prompts) | ~50K prompts |
| Eval | WikiText-103 | test split |
| Baseline | OpenAI WebGPT Comparisons | secondary |

---

## Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| RM Accuracy | Preference prediction accuracy (held-out) | 69–72% |
| Win Rate (RLHF vs SFT) | RM-scored response win rate | > 55% |
| Win Rate (RLHF vs BoN) | vs. Best-of-N sampling baseline | > 50% |
| KL Divergence | Per-step policy drift from SFT | < 10 nats |
| WikiText-103 PPL | Language modeling retention | ≈ SFT ± 5 |

---

## Novel Ablation

**Research Question:** Does an undersized RM (2× fewer params than policy) degrade alignment?

This experiment is **not explicitly reported** in:
- Ouyang et al. (InstructGPT, NeurIPS 2022)  
- Bai et al. (Anthropic, 2022)

We test the under-examined assumption that RM capacity must match policy capacity.

Run the ablation in `02_reward_model.ipynb` (set `RUN_ABLATION = True`).

---

## References

1. Christiano et al. (2017). *Deep Reinforcement Learning from Human Preferences*. NeurIPS.
2. Ouyang et al. (2022). *Training language models to follow instructions with human feedback*. NeurIPS.
3. Bai et al. (2022). *Training a Helpful and Harmless Assistant with RLHF*. Anthropic.
4. Stiennon et al. (2020). *Learning to summarize with human feedback*. NeurIPS.

---

## Member Contributions

| Member | Responsibilities |
|--------|-----------------|
| Kumi Acheampong | SFT implementation, PPO fine-tuning, evaluation pipeline |
| Selase Doku | Reward model training, dataset preprocessing, literature review & writeup |