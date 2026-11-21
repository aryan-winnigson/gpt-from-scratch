# Stage 06 — Multi-head attention

> **Status:** not written yet. We'll build this lesson together when you get here, so it
> can be shaped around the questions you're actually asking by then.

Many heads in parallel, in one batched matmul.

## What you'll build

- CausalSelfAttention with nh heads and an output projection
- Verified identical to nh separate single heads concatenated
- Verified identical to F.scaled_dot_product_attention

## Why it matters

Where most from-scratch implementations silently break. The split_heads drill from Stage 00 is the reason yours won't.
