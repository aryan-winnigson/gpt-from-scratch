# GPT From Scratch

Building a GPT from first principles — every line written by hand, with the understanding
to rebuild it anywhere, on any machine, without reference.

**Author:** Aryan Rawat
**Started:** 21 November 2026

---

## The goal

By the end of this repo I will have:

1. A **scalar autograd engine** I wrote, that I understand well enough to re-derive.
2. A **tokenizer** (BPE) I trained on my own corpus.
3. A **GPT** — attention, multi-head, blocks, residuals, the whole architecture — in
   `mygpt/`, written by me, that can load the real GPT-2 weights and reproduce OpenAI's
   output token for token. (If my architecture is wrong, this test fails. It is the proof.)
4. **My own trained model**, from scratch, on my own data.
5. The ability to **explain and rebuild any piece of it** from memory.

Output quality is explicitly *not* the goal. Correct architecture, understood deeply, is.

---

## How this repo works

Each stage lives in `stages/NN_name/` and contains a `README.md` with the theory, the
exercises, and the questions I should be able to answer before moving on.

The **code I write does not live in the stage folder** — it lives in the real packages
(`micrograd/`, `mygpt/`). Stages are the lesson; the packages are the product. By the end,
`mygpt/` is a genuine, working GPT library with my name on it.

Every stage ships with a **pytest suite that checks my implementation against PyTorch's
built-in equivalent**. If `tests/test_05_attention.py` passes, my attention is numerically
identical to `F.scaled_dot_product_attention`. That's the feedback loop: no one tells me
the answer, the tests tell me when I've found it.

### The loop for every stage

```bash
# 1. Read the lesson
open stages/05_attention/README.md

# 2. Write the code in the file it points at (mygpt/attention.py)

# 3. Run the tests until they pass
uv run pytest tests/test_05_attention.py -v

# 4. Write down what I learned, in my own words
open notes/05_attention.md

# 5. Commit — the git history IS the learning record
git add -A && git commit -m "stage 05: single-head causal self-attention"
git tag stage-05 && git push --tags
```

Rule I'm holding myself to: **I do not move to stage N+1 until I can answer every question
at the bottom of stage N's README without looking anything up.**

---

## Setup

```bash
uv sync                                  # Python 3.12 + torch + friends
uv run python -c "import torch; print(torch.backends.mps.is_available())"   # expect: True
uv run pytest                            # run everything written so far
```

Hardware: Apple M5, 24 GB unified memory. Local training uses the **MPS** backend for
everything up to ~30M parameters. Anything bigger goes to a free Colab T4/A100 — see
`stages/11_colab_training/`.

---

## Syllabus

Fifteen stages, four acts. Check them off as they land.

### Act I — Foundations (what a neural net actually is)

| | Stage | What I build | Why it matters |
|-|-------|--------------|----------------|
| [ ] | **00** · [Setup & tensor fluency](stages/00_setup/) | Env check, tensor/broadcasting exercises, cross-entropy by hand | Every bug for the next 3 months will be a shape bug. Fluency here is not optional. |
| [ ] | **01** · [Autograd from scratch](stages/01_autograd/) | `micrograd/` — a scalar `Value` class with `.backward()`, then an MLP trained on it | Backprop stops being magic. This is the single highest-leverage stage in the repo. |
| [ ] | **02** · [Tokenization](stages/02_tokenization/) | Char-level codec, then a real BPE trainer + encoder/decoder | Text → integers. Every weird LLM behaviour (spelling, arithmetic, non-English) traces back here. |
| [ ] | **03** · [Bigram language model](stages/03_bigram/) | The simplest possible neural LM + the full training loop skeleton | Establishes the scaffolding — data batching, loss, eval, sampling — that every later stage reuses. |

### Act II — Attention (the actual idea)

| | Stage | What I build | Why it matters |
|-|-------|--------------|----------------|
| [ ] | **04** · [Core layers](stages/04_layers/) | `Linear`, `Embedding`, `softmax`, cross-entropy, `LayerNorm`, `GELU`, `Dropout` — plus two backward passes derived by hand | These are the Lego bricks. Hand-deriving LayerNorm's backward is where the chain rule becomes muscle memory. |
| [ ] | **05** · [Self-attention](stages/05_attention/) | One causal head, built in 7 steps from "average the past tokens" to scaled dot-product | The core insight of the whole field. Earn it slowly. |
| [ ] | **06** · [Multi-head attention](stages/06_multihead/) | Batched heads, the reshape/transpose dance, output projection | Where most from-scratch implementations silently break. The tests catch it. |

### Act III — The model

| | Stage | What I build | Why it matters |
|-|-------|--------------|----------------|
| [ ] | **07** · [The Transformer block](stages/07_block/) | FFN, residual stream, pre-LN vs post-LN | Why GPT-2 trains at all and the original 2017 Transformer needed a warmup babysitter. |
| [ ] | **08** · [The full GPT](stages/08_gpt/) | Positional embeddings, weight tying, init scheme, `GPT` class — **then load real GPT-2 124M weights into it** | The proof. If OpenAI's weights produce OpenAI's text inside my code, my architecture is correct. |
| [ ] | **09** · [Optimization](stages/09_optimizer/) | AdamW from scratch, warmup + cosine schedule, grad clipping, weight-decay groups, grad accumulation, checkpointing | The difference between "a model" and "a model that learns". |

### Act IV — Making it real

| | Stage | What I build | Why it matters |
|-|-------|--------------|----------------|
| [ ] | **10** · [Training locally](stages/10_train_local/) | Overfit-one-batch sanity check, then TinyShakespeare on MPS, loss curves | First time my own weights generate my own text. |
| [ ] | **11** · [Training on Colab](stages/11_colab_training/) | memmap data pipeline, bf16/AMP, `torch.compile`, TinyStories on a free GPU | Scaling up on hardware I don't own, for free. |
| [ ] | **12** · [Inference & KV cache](stages/12_inference/) | Temperature, top-k, top-p, and a KV cache with benchmarks | Understanding the cache is understanding what attention actually stores. |
| [ ] | **13** · [Modern GPT](stages/13_modern/) | RoPE, RMSNorm, SwiGLU, GQA, FlashAttention | Upgrade my GPT-2 into a Llama-class model. This is what 2026 architectures actually ship. |
| [ ] | **14** · [Capstone](stages/14_capstone/) | My trained model + model card + a written explanation of every component | If I can write the explanation, I own the knowledge. |

---

## Repo layout

```
micrograd/        Stage 01. My scalar autograd engine.
mygpt/            Stages 02-13. My GPT library. The product.
stages/           The lessons. Theory, exercises, and the questions I must answer.
tests/            Correctness checks against PyTorch ground truth.
notes/            My own explanations, in my own words. Written after each stage.
scripts/          Data download, training entrypoints, benchmarks.
data/             Corpora and tokenized binaries (gitignored — regenerate with scripts/).
colab/            Notebooks for the free-GPU training runs.
```

## Progress log

| Date | Stage | Note |
|------|-------|------|
| 2026-11-21 | — | Repo created. Let's go. |
