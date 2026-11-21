# Stage 05 — Self-attention

> **Status:** not written yet. We'll build this lesson together when you get here, so it
> can be shaped around the questions you're actually asking by then.

One causal head, built in seven steps.

## What you'll build

- Step 1: average the past with a loop (you already did this in Stage 00)
- Step 2: the same thing as a matrix multiply with a triangular matrix
- Step 3: the same thing again via masked softmax
- Step 4-7: learned queries, keys, values; the 1/sqrt(d_k) scale; the causal mask

## Why it matters

The core idea of the entire field. Each step is a small change to the one before, so nothing is a leap.
