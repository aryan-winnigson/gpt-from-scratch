# Stage 09 — Optimization

> **Status:** not written yet. We'll build this lesson together when you get here, so it
> can be shaped around the questions you're actually asking by then.

Making it actually learn.

## What you'll build

- AdamW implemented from scratch, checked against torch.optim.AdamW
- Warmup + cosine decay schedule; gradient clipping
- Weight-decay parameter groups (and why biases and LayerNorms are excluded)
- Gradient accumulation and checkpointing

## Why it matters

The difference between 'a model' and 'a model that learns'.
