# Stage 11 — Training on Colab

> **Status:** not written yet. We'll build this lesson together when you get here, so it
> can be shaped around the questions you're actually asking by then.

Scaling up on a free GPU.

## What you'll build

- memmap .bin data pipeline (no more loading the dataset into RAM)
- bf16 autocast, torch.compile, and measuring the actual speedup
- A bigger model on TinyStories; checkpoints saved to Drive
- Reading the loss curve: is it compute-bound, data-bound, or broken?

## Why it matters

Free T4/A100 time is enough for a model that writes real sentences.
