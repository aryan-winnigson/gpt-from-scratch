# Stage 12 — Inference & the KV cache

> **Status:** not written yet. We'll build this lesson together when you get here, so it
> can be shaped around the questions you're actually asking by then.

Making generation fast and controllable.

## What you'll build

- Greedy, temperature, top-k, top-p (nucleus) sampling
- A KV cache, with before/after tokens-per-second benchmarks
- Verified: cached generation is token-identical to uncached

## Why it matters

Understanding the cache is understanding what attention actually stores — and why context length costs so much memory.
