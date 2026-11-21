# Stage 02 — Tokenization

> **Status:** not written yet. We'll build this lesson together when you get here, so it
> can be shaped around the questions you're actually asking by then.

Text to integers and back.

## What you'll build

- Char-level encoder/decoder on TinyShakespeare
- A BPE trainer: merge the most frequent adjacent pair, repeatedly
- encode() / decode() that round-trip perfectly, including bytes outside ASCII
- Compare your tokenizer's compression ratio against tiktoken's gpt2 encoding

## Why it matters

Why GPT can't spell 'strawberry', why it's bad at arithmetic, and why non-English text costs more — all three trace back to this stage.
