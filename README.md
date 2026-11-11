# Build an LLM from Scratch: Training Dynamics and Architecture Experiments

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/llm-from-scratch/blob/main/LLM_Scratch_Experiments.ipynb)

We train a small GPT language model from scratch in PyTorch on a Project Gutenberg book, then study how its
training and text generation change when we vary:

- the **number of epochs and the learning rate**,
- the **number of transformer layers**,
- the **number of attention heads**, and
- which **components** the model has (LayerNorm, shortcut connections, feed-forward network).

This is the code and results for the *Build LLM: Research-Based Assessment*. The full write-up is in
[`report/LLM_Scratch_Report.pdf`](report/LLM_Scratch_Report.pdf).

---

## Repository structure

```
.
├── LLM_Scratch_Experiments.ipynb   # main notebook: data, model, training engine, all experiments, export
├── results/                        # outputs produced by the notebook
│   ├── figures/                    # every plot (PDF + PNG)
│   ├── tables/                     # summary tables (CSV + LaTeX)
│   ├── runs/                       # one JSON log per training run (losses, gradient norms, samples)
│   ├── samples.md                  # all generated text
│   ├── dataset_stats.json
│   ├── config_and_environment.json
│   └── runs_index.json             # run label -> JSON file
└── report/                         # Overleaf (LaTeX) source of the report + compiled PDF
    ├── main.tex
    ├── figures/ generated/ code/
    └── LLM_Scratch_Report.pdf
```

---

## Setup

| | |
|---|---|
| **Dataset** | *The Strange Case of Dr Jekyll and Mr Hyde*, R. L. Stevenson ([Project Gutenberg #43](https://www.gutenberg.org/ebooks/43)): 33,822 training and 3,443 validation tokens (90/10 split) |
| **Tokenizer** | GPT-2 BPE (`tiktoken`), vocabulary 50,257 |
| **Base model** | GPT, 6 layers, 4 heads, 256-dim embeddings, context 256, dropout 0.1: 30.5M parameters (4.7M outside the embeddings) |
| **Training** | AdamW, lr 4e-4, weight decay 0.1, batch size 8, 30 epochs, fp32, seed 123 |
| **Hardware** | One NVIDIA L4 GPU (Google Colab), about 40 minutes for all 17 runs |

The model follows the GPT architecture built in class: token and positional embeddings, pre-LayerNorm
transformer blocks with causal multi-head attention and a GELU feed-forward network, and a final LayerNorm
and output head. It is scaled down from the 124M GPT-2 configuration so that 500-epoch runs fit in Colab.
Three flags in the config turn components off for the ablations: `use_norm`, `use_shortcut` and `use_ffn`.

---

## Experiments and key findings

| Experiment | Settings | Main finding |
|---|---|---|
| **Baseline** | settings above | Best validation loss **6.25** at epoch 12, then overfits (validation 7.25, training 1.21 by epoch 30) |
| **1. Epochs × learning rate** | lr 1e-4 / 1e-3 / 1e-2, checkpoints at 5–500 epochs | **1e-3** is best (validation 6.24 at epoch 7). 1e-4 is about 5× slower. 1e-2 is unstable (gradient spikes over 100). By 500 epochs every run has memorised the book (up to 94% of generated 8-grams copied word for word) |
| **2. Layers** | 1, 3, 5, 7, 12 | Depth lowers training loss (2.85 → 0.69) and makes overfitting start earlier. Best validation loss barely moves (6.32 → 6.23) |
| **3. Attention heads** | 1, 2, 4, 8 | No meaningful difference (best validation 6.246–6.253; same parameter count) |
| **4. Ablations** | no LayerNorm / no shortcut / no FFN | Without LayerNorm or shortcuts the model is **stuck at the token-frequency level** (loss about 6.1) and outputs only newlines. Without the FFN it still learns, but more slowly and with more repetition |

**Takeaway:** on a single short book, the data limits how well any model can generalise. The learning rate,
early stopping, and the residual and normalisation path decide whether and how fast a model reaches that
limit. Extra depth and heads mostly make the model memorise faster.

<p align="center">
  <img src="results/figures/exp1_grid_logx.png" width="100%"><br>
  <em>Experiment 1: training (solid) and validation (dashed) loss per learning rate, log-scale epochs.</em>
</p>

<p align="center">
  <img src="results/figures/exp4_step_loss.png" width="70%"><br>
  <em>Experiment 4: removing LayerNorm or the shortcut connections stops learning at the token-frequency plateau.</em>
</p>

---

## How to reproduce

1. Open `LLM_Scratch_Experiments.ipynb` in Google Colab (badge above) and choose
   **Runtime → Change runtime type → GPU** (a T4 or L4 is enough).
2. Optional smoke test: set `QUICK_TEST = True` in the configuration cell and run all cells (about 2–3 minutes).
3. Set `QUICK_TEST = False` and run all cells. The notebook:
   - downloads and cleans the book from Project Gutenberg,
   - prints a time estimate for each experiment on your GPU,
   - runs the baseline and Experiments 1–4,
   - saves figures, tables, run logs and samples to `llm_scratch_results/` (or Google Drive if `USE_DRIVE = True`),
   - zips everything at the end.

Finished runs are cached by a hash of their configuration, so re-running after a Colab disconnect skips completed work.

### Configuration

All settings are in one cell near the top of the notebook:

| Setting | Default | Meaning |
|---|---|---|
| `BOOK_IDS` | `[43]` | Gutenberg book IDs (several are combined) |
| `BASE_MODEL_CFG` | 6 layers, 4 heads, 256 dims | base architecture and ablation flags |
| `BASE_TRAIN_CFG` | lr 4e-4, 30 epochs, batch 8 | base training settings |
| `EXP1_LRS`, `EXP1_EPOCH_CHECKPOINTS` | `[1e-4, 1e-3, 1e-2]`, `[5 … 500]` | Experiment 1 grid |
| `EXP2_LAYERS`, `EXP3_HEADS` | `[1,3,5,7,12]`, `[1,2,4,8]` | Experiments 2 and 3 |
| `EXP4_ABLATIONS` | no norm / no shortcut / no FFN | Experiment 4 |
| `USE_AMP` | `False` | fp16 mixed precision (faster, slightly less exact) |

**Running locally:** Python 3.10+ and `pip install torch tiktoken numpy pandas matplotlib jinja2`, then open the
notebook in Jupyter. A GPU is strongly recommended.

### Building the report

Upload the `report/` folder to [Overleaf](https://www.overleaf.com) (New Project → Upload Project) and compile
with pdfLaTeX, or run `latexmk -pdf main.tex` locally.

---

## Metrics recorded per run

- Training and validation cross-entropy after every epoch (dropout off), plus the loss of every training batch
- Mean gradient L2 norm per epoch
- Generated text for two prompts, with greedy decoding and top-k sampling (k = 25, T = 1)
- **Distinct-2:** unique bigrams ÷ all bigrams in a sample; low means repetitive text
- **Copy rate:** share of generated 8-grams found word for word in the training text; high means memorisation

## Limitations

Each configuration was trained once (one seed), and the validation set has only 13 sequences, so
differences in validation loss below about 0.02–0.03 are within noise. The model is a scaled-down GPT-2 and
the corpus is a single short book.

## Acknowledgements

- The model and training code build on the course notebook, which follows Sebastian Raschka's
  *Build a Large Language Model (From Scratch)* ([rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)).
- The text is from [Project Gutenberg](https://www.gutenberg.org).

## Team

- Siju George
