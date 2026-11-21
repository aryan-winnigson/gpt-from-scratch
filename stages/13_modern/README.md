# Stage 13 — Modern GPT

> **Status:** not written yet. We'll build this lesson together when you get here, so it
> can be shaped around the questions you're actually asking by then.

Upgrade GPT-2 (2019) to Llama-class (2026).

## What you'll build

- RoPE instead of learned positional embeddings
- RMSNorm instead of LayerNorm; SwiGLU instead of GELU MLP
- Grouped-query attention; FlashAttention via scaled_dot_product_attention
- Ablate each one and measure what it actually buys you

## Why it matters

These four changes are what separates a 2019 architecture from what ships today.
