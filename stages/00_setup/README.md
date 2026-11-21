# Stage 00 — Setup & Tensor Fluency

> **Time:** 1–2 hours
> **You write:** `stages/00_setup/tensor_drills.py`
> **You pass:** `uv run pytest tests/test_00_setup.py -v`

---

## Why this stage exists

You are about to spend many hours debugging a neural network. Here is a prediction: **the
overwhelming majority of your bugs will be shape bugs and broadcasting bugs**, not
conceptual ones. You will stare at `RuntimeError: shapes (4, 8, 32) and (8, 32, 4) cannot
be multiplied` and lose an afternoon.

So before any machine learning, we buy fluency. Seven small drills. Each one is a thing
you will literally use later — I've noted where. Nothing here is busywork.

---

## 1. What is a language model, really?

Strip away everything and a language model is one function:

$$p(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

"Given the tokens so far, what's the probability distribution over the next one?"

That's it. Not "understanding", not "reasoning" — a conditional probability distribution
over a finite vocabulary. GPT-2 has a vocabulary of 50,257 tokens, so for every position
it outputs 50,257 numbers that sum to 1.

Everything in this repo serves that one function. Attention is a way to compute it.
Transformers are a way to stack that computation. Training is fitting it to data.

### How we score it

The model outputs a probability for the token that actually came next. A good model gives
that token high probability. So we maximise the probability of the true continuation —
or equivalently, minimise the **negative log likelihood**:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log p(\text{correct token}_i)$$

This is **cross-entropy loss**. Drill 3 makes you build it.

Two numbers worth internalising, for a vocabulary of size $V$:

- A model that outputs a **uniform** distribution — pure ignorance — gets loss
  $-\log(1/V) = \log V$. For char-level Shakespeare ($V = 65$) that's **4.17**. For GPT-2
  ($V = 50257$) that's **10.82**. *This is your sanity baseline.* On step 0 of training,
  before any learning, your loss should be almost exactly $\log V$. If it isn't, your
  initialisation is wrong — and you've caught the bug in 10 seconds instead of 10 hours.
- **Perplexity** = $e^{\mathcal{L}}$. It's the loss expressed as "the model is as confused
  as if it were picking uniformly among this many options." Loss 4.17 → perplexity 65
  ("no idea"). Loss 1.5 → perplexity 4.5 ("it's narrowed down to about 4 characters").

---

## 2. The three tensor mechanics you must own

### Broadcasting

When shapes don't match, PyTorch aligns them **from the right** and stretches any
dimension of size 1:

```
(4, 8, 32)      (B, T, C)  — a batch of 4 sequences, 8 tokens each, 32 numbers per token
  (8, 32)       aligns to (1, 8, 32), stretched over the batch   ✅
  (4, 1)        aligns to (1, 4, 1)   — 4 ≠ 8 on the middle axis  ❌
(4, 8, 1)       stretches over C — "one number per token"         ✅
```

The dangerous case is the one that *doesn't* error. `(8, 1)` and `(1, 8)` broadcast
happily into `(8, 8)` — silently, and almost never what you meant. When something is
numerically wrong rather than crashing, suspect this first.

### `keepdim`

```python
x.sum(dim=-1)                 # (4, 8, 32) -> (4, 8)      lost the axis
x.sum(dim=-1, keepdim=True)   # (4, 8, 32) -> (4, 8, 1)   kept it, as size 1
```

You want `keepdim=True` almost every time you're about to divide, because `(4,8,32) /
(4,8,1)` broadcasts correctly while `(4,8,32) / (4,8)` explodes. Softmax and LayerNorm
both live or die on this.

### Batched matmul

`@` multiplies the **last two dimensions** and broadcasts everything to the left:

```
(B, T, C) @ (C, H)        -> (B, T, H)     a Linear layer applied to every token
(B, T, T) @ (B, T, C)     -> (B, T, C)     attention weights applied to values
(B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)   every head's attention scores at once
```

Read those three lines again at the end of Stage 06. They're the entire shape story of a
transformer.

### Numerical stability (the softmax trick)

`exp(1000)` is `inf`, and `inf/inf` is `nan`. Once a `nan` enters your network it spreads
to every weight on the next backward pass and the run is dead. The fix exploits the fact
that softmax is invariant to shifts:

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - c}}{\sum_j e^{x_j - c}} \quad \text{for any } c$$

Choose $c = \max_j x_j$. Now the largest exponent is $e^0 = 1$ and nothing can overflow.
Every real softmax implementation on earth does this. Yours will too, in Drill 2.

> **Convince yourself the identity is true** — multiply top and bottom by $e^{-c}$. If you
> can't see it in 30 seconds, write it out. Small algebra now, deep comfort later.

---

## 3. The drills

Open `stages/00_setup/tensor_drills.py`. Seven functions, all stubbed with docstrings and
shape contracts. Fill them in.

| # | Function | Used later in |
|---|----------|---------------|
| 1 | `one_hot` | Stage 03 — understanding what an embedding lookup *is* |
| 2 | `stable_softmax` | Stage 04, 05 — attention weights |
| 3 | `cross_entropy` | every training loop in this repo |
| 4 | `gather_target_probs` | the indexing pattern inside cross-entropy |
| 5 | `causal_mask` | Stage 05 — the mask that makes GPT autoregressive |
| 6 | `split_heads` | Stage 06 — *the* reshape that breaks most implementations |
| 7 | `mean_of_past` | Stage 05 — this is literally attention, step one |

**Rules:**

- Pure PyTorch tensor ops only. The banned list is in each docstring — it's always the
  one function that would do the whole drill for you (`F.softmax`, `F.one_hot`, …).
- No Python `for` loops except in `mean_of_past`, where the loop *is* the point.
- Don't peek at `tests/test_00_setup.py` before trying. Run it, read the failure, fix.

### Getting unstuck

Run the tests and read the shapes in the error. If you're stuck for more than ~15 minutes
on a single drill, ask me — but ask a *specific* question ("why does `keepdim` matter
here?" rather than "this doesn't work"). Specific questions are how you find out what you
actually don't know.

---

## ✅ Before you move to Stage 01

Answer these out loud, in your own words, without looking:

1. A language model is a function. What are its inputs and what are its outputs?
2. Your char-level model has a vocabulary of 65. Training starts and step 0 prints loss
   `4.17`. Is that good, bad, or expected — and how do you know?
3. Same model, step 0 prints loss `11.4`. What does that tell you, and where do you look?
4. Why does `stable_softmax` subtract the max? What breaks if you don't?
5. You have `x` of shape `(4, 8, 32)` and want to divide each token's vector by its own
   sum. Write the one line. Why does the version without `keepdim=True` fail?
6. What does `(B, nh, T, hs) @ (B, nh, hs, T)` produce, and what does each number in the
   result *mean*?
7. In `mean_of_past`, position 5's output averages positions 0 through 5. Why must it
   never include position 6?

Then:

```bash
uv run pytest tests/test_00_setup.py -v
# write notes/00_setup.md in your own words
git add -A && git commit -m "stage 00: tensor fluency drills"
git tag stage-00 && git push && git push --tags
```
