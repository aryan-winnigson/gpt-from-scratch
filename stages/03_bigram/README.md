# Stage 03 — Bigram language model

> **Status:** not written yet. We'll build this lesson together when you get here, so it
> can be shaped around the questions you're actually asking by then.

The simplest neural LM, and the training loop everything else reuses.

## What you'll build

- A (V, V) embedding table as the whole model
- Batch sampling: get_batch() producing (B, T) inputs and targets
- Training loop with train/val split and periodic eval
- generate() — sample from the model autoregressively

## Why it matters

Establishes the scaffolding: data batching, loss, evaluation, sampling. Stages 05-08 only replace the middle.
