# Stage 04 — Core layers

> **Status:** not written yet. We'll build this lesson together when you get here, so it
> can be shaped around the questions you're actually asking by then.

The Lego bricks, each tested against torch.nn.

## What you'll build

- Linear, Embedding (with proper init)
- softmax and cross-entropy on tensors
- LayerNorm — forward, then the backward pass derived BY HAND
- GELU (exact and the tanh approximation GPT-2 actually uses), Dropout

## Why it matters

Hand-deriving LayerNorm's backward is the hardest maths in this repo and the point where the chain rule becomes muscle memory.
