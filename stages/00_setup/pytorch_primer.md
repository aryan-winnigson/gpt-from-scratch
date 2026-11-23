# PyTorch, in just enough depth to start

> **Time:** 45–60 minutes, with the REPL open.
> **Prerequisite for:** `tensor_drills.py`
> **Scope:** the ~20 functions the drills need. Nothing else.

PyTorch is enormous and you need almost none of it. This repo touches a small, stable
core, and that core is what's below. If you know numpy, 90% of this is a renaming
exercise and you can skim to §6.

**Read this with a REPL open.** Reading about tensors does not work; typing them does.

```bash
cd /Users/aryanrawat/learning/gpt
uv run python
```

Type every example. Then change a number and type it again.

---

## 0. The one mental model

A tensor is **a flat array of numbers plus a shape that tells you how to read it.**

That's it. `torch.arange(12)` is twelve numbers in a row. Giving it shape `(3, 4)`
doesn't move anything in memory — it just says "read these twelve as 3 groups of 4."
Giving it shape `(2, 2, 3)` says "read them as 2 groups of 2 groups of 3." Same twelve
numbers, three different interpretations.

Hold onto this. It's the difference between memorising `.view()` and understanding it,
and it's the entire explanation of the trap in drill 6.

A tensor also has a **dtype** (what kind of number) and a **device** (CPU/MPS/GPU). Both
will bite you exactly once each, so they get a section.

---

## 1. Making tensors

```python
import torch

torch.tensor([1, 2, 3])        # tensor([1, 2, 3])        from a Python list
torch.tensor([1.0, 2.0])       # tensor([1., 2.])
torch.zeros(2, 3)              # 2x3 of 0.0
torch.ones(2, 3)               # 2x3 of 1.0
torch.arange(5)                # tensor([0, 1, 2, 3, 4])   like range(), as a tensor
torch.randn(2, 3)              # 2x3 from a standard normal (mean 0, std 1)
torch.randint(0, 10, (4,))     # 4 random ints in [0, 10)
torch.zeros_like(x)            # zeros with the same shape AND dtype as x
```

Note the inconsistency: `zeros(2, 3)` takes loose arguments, `randint` takes the shape as
a **tuple**. This is just how it is; you'll misremember it occasionally forever.

`torch.manual_seed(1337)` makes the random ones reproducible. The tests do this so your
results are identical to mine.

### The dtype gotcha

```python
torch.tensor([1, 2, 3]).dtype    # torch.int64    ← no decimal point, so integers
torch.tensor([1., 2., 3.]).dtype # torch.float32  ← decimal point, so floats
torch.zeros(2, 3).dtype          # torch.float32
torch.arange(5).dtype            # torch.int64
```

Two rules that cover every dtype problem you'll hit in this repo:

1. **Anything that gets multiplied or averaged must be float.** Integer tensors refuse to
   compute a mean at all:
   ```python
   torch.arange(12).reshape(3, 4).mean()
   # RuntimeError: mean(): could not infer output dtype. Input dtype must be either a
   # floating point or complex dtype.
   ```
   Fix with `.float()`. That error message is a dtype problem in disguise — now you'll
   recognise it in one second instead of ten minutes.

2. **Anything used as an index must be int64** (PyTorch calls it `long`). Token IDs,
   target classes, positions. `torch.arange` already gives you int64, which is convenient
   and not an accident.

---

## 2. Shape is the thing you debug

```python
x = torch.randn(2, 3, 4)
x.shape        # torch.Size([2, 3, 4])   — behaves like a tuple
x.shape[0]     # 2
x.ndim         # 3     how many dimensions
x.numel()      # 24    how many numbers in total
B, T, C = x.shape      # unpack it like a tuple. You'll write this line constantly.
```

**Build the habit now:** when something is wrong, print shapes before you print values.

```python
print(f"{x.shape=} {y.shape=}")     # the = inside an f-string prints the name too
```

Nine failures out of ten, the shapes tell you the answer immediately.

---

## 3. Indexing, part 1: slicing

Identical to numpy, and close to Python lists with more commas.

```python
x = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

x[1]        # tensor([4, 5, 6, 7])      row 1
x[1, 2]     # tensor(6)                 row 1, col 2 — a 0-dim tensor, not a Python int
x[:, 2]     # tensor([ 2,  6, 10])      every row, column 2
x[0:2]      # the first two rows
x[:, 1:3]   # every row, columns 1 and 2
x[-1]       # the last row
```

The comma separates **dimensions**; the colon means **all of this dimension**.
`x[:, 2]` is "all rows, column 2".

Two more you'll use:

```python
x[..., -1]   # `...` means "all the leading dimensions, however many there are".
             # Handy when you want the last dim and don't know the rank.
x[None]      # inserts a new dimension of size 1 at the front: (3,4) -> (1,3,4)
```

Slicing returns a **view**, not a copy — it shares memory with the original. So
`x[0] += 1` modifies `x`. Use `.clone()` when you want a genuine copy.

---

## 4. Indexing, part 2: indexing *with tensors*

This is the part that has no equivalent in plain Python, it's used in five of the seven
drills, and it's where the real power is. Two cases.

**One index tensor → selects rows, in whatever order you ask, repeats allowed:**

```python
x[torch.tensor([2, 0])]
# tensor([[ 8,  9, 10, 11],
#         [ 0,  1,  2,  3]])        row 2, then row 0
```

This is exactly what an embedding lookup is. `emb[token_ids]` with 1000 token IDs gives
you 1000 rows in one operation. Remember that in Stage 03.

**Two index tensors → paired, element by element:**

```python
rows = torch.tensor([0, 1, 2])
cols = torch.tensor([3, 2, 1])
x[rows, cols]
# tensor([3, 6, 9])
```

Read it carefully: you did **not** get a 3×3 block. You got
`[x[0,3], x[1,2], x[2,1]]` — the tensors are **zipped**, giving one number per pair. The
output has the shape of the index tensors, not of `x`.

> This is the single most useful indexing pattern in ML code, and the one people
> consistently get wrong by expecting a cross-product. When you need "one specific
> element from each row", this is the tool.

**Indexing also works on the left of an `=`:**

```python
a = torch.zeros(3, 4)
a[torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])] = 1.0
# tensor([[0., 1., 0., 0.],
#         [0., 0., 1., 0.],
#         [0., 0., 0., 1.]])
```

**Boolean masks** work too — index with a bool tensor of the same shape:

```python
x[x > 6]                 # tensor([ 7,  8,  9, 10, 11])  — flattened, only the True spots
x.masked_fill(x > 6, 0)  # replace where True, keep the shape. You'll use this in Stage 05.
```

---

## 5. Elementwise maths and broadcasting

Every operator works elementwise on matching shapes:

```python
a + b,  a - b,  a * b,  a / b,  a ** 2
a.exp(),  a.log(),  a.sqrt(),  a.abs(),  a.tanh()
torch.exp(a)      # identical to a.exp() — both spellings exist for most functions
```

> `a * b` is **elementwise** multiplication. Matrix multiplication is `a @ b`. Confusing
> these is a rite of passage; the shapes usually catch it.

### Broadcasting

When shapes differ, PyTorch aligns them **from the right** and stretches any dimension of
size 1:

```python
torch.ones(3, 4) / torch.tensor([1., 2., 4., 8.])     # (3,4) / (4,) -> fine, (4,) aligns with the last dim
torch.ones(3, 4) / torch.ones(3)                      # (3,4) / (3,) -> ERROR
# RuntimeError: The size of tensor a (4) must match the size of tensor b (3)
#               at non-singleton dimension 1
```

The second one fails because alignment is from the **right**: the `3` lines up against
the `4`, not against the other `3`. To divide each *row* by its own number you need shape
`(3, 1)`, not `(3,)`. Which is the entire reason `keepdim` exists — next section.

The dangerous case is the one that doesn't error: `(4,1)` and `(1,4)` broadcast silently
into `(4,4)`. If a result is numerically wrong but didn't crash, suspect this first.

---

## 6. Reductions, and `keepdim`

```python
x = torch.arange(12).reshape(3, 4).float()

x.sum()                      # tensor(66.)     everything, down to a scalar
x.sum(dim=1)                 # tensor([ 6., 22., 38.])   sum along dim 1 → shape (3,)
x.sum(dim=1, keepdim=True)   # shape (3, 1)              same numbers, dimension kept
```

`dim=k` means **"collapse dimension k"**. The result loses that dimension — unless
`keepdim=True`, which leaves it behind as size 1.

Why that matters, concretely:

```python
x / x.sum(dim=1, keepdim=True)   # (3,4) / (3,1) → each row divided by its own sum ✅
x / x.sum(dim=1)                 # (3,4) / (3,)  → RuntimeError, as in §5          ❌
```

**Rule of thumb: if you're about to divide by a reduction, you want `keepdim=True`.**
Softmax, LayerNorm, and attention all depend on this.

`dim=-1` means the last dimension. Prefer it over hardcoding `dim=2` — it keeps your code
working when a batch dimension appears later. Everything in this repo reduces over the
last dim.

### The `max` gotcha

```python
x.max()                 # tensor(11.)   no dim → plain scalar
m = x.max(dim=1)        # WITH a dim → a named tuple of TWO things
m.values                # tensor([ 3.,  7., 11.])
m.indices               # tensor([3, 3, 3])      where each max was found
x.max(dim=1).values     # what you usually want
x.max(dim=1, keepdim=True).values   # shape (3,1) — this is the softmax one
```

`.max(dim=...)` returning a pair surprises everyone once. `argmax` gives you just the
indices, and is what you'll use for greedy decoding in Stage 12.

Also available: `.mean()`, `.min()`, `.std()`, `.cumsum()`, `.any()`, `.all()` — all take
`dim` and `keepdim` the same way.

---

## 7. Changing shape

### `.view()` and `.reshape()`

```python
x = torch.arange(12)
x.view(3, 4)       # read the same 12 numbers as 3 rows of 4
x.view(2, 2, 3)    # read them as 2 x 2 x 3
x.view(-1, 4)      # -1 means "work it out" → (3, 4)
```

The filling order is **row-major**: the **last dimension varies fastest**. `arange(12)`
viewed as `(3,4)` puts `0,1,2,3` in the first row, because you fill along the last axis
before stepping the one before it.

> Internalise that sentence. Drill 6's trap is a direct consequence of it, and no amount
> of permuting-until-green will substitute for it. When you need to predict what a
> `.view()` does, ask: *which axis varies fastest, and is that the grouping I want?*

### `.transpose()` and `.permute()`

```python
y = torch.arange(24).reshape(2, 3, 4)
y.transpose(1, 2).shape    # (2, 4, 3)     swap exactly two dimensions
y.permute(2, 0, 1).shape   # (4, 2, 3)     reorder all of them at once
```

`transpose` genuinely rearranges which number sits where. `view` never does — it only
reinterprets. **That difference is the whole of drill 6.**

### Contiguity (the error you will hit)

```python
t = y.transpose(1, 2)
t.is_contiguous()    # False — transpose returns a view with shuffled strides
t.view(2, 12)
# RuntimeError: view size is not compatible with input tensor's size and stride...
#               Use .reshape(...) instead.
t.reshape(2, 12)     # works — copies if it has to
t.contiguous().view(2, 12)   # also works, and makes the copy explicit
```

`view` demands the memory already be laid out right; `reshape` copies when it isn't. Real
transformer code says `.transpose(1,2).contiguous().view(B, T, C)` for exactly this
reason, and you'll write that line yourself in Stage 06.

Three more:

```python
x.unsqueeze(0)    # add a size-1 dimension at position 0
x.squeeze(0)      # remove a size-1 dimension
x.flatten()       # collapse to 1-D
```

---

## 8. Matrix multiplication

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)
A @ B                        # (3,5)   inner dimensions must match: 4 == 4
```

With more than 2 dimensions, `@` multiplies the **last two** and broadcasts the rest:

```python
torch.randn(2, 3, 4) @ torch.randn(4, 5)              # (2, 3, 5)
torch.randn(8, 2, 3, 4) @ torch.randn(8, 2, 4, 5)     # (8, 2, 3, 5)
```

This is how a transformer does every head of every sequence in the batch in one call.
When you see `(B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)` in Stage 06, it's just
this rule: `B` and `nh` come along for the ride, `T×hs` times `hs×T` gives `T×T`.

---

## 9. Scalars, printing, comparing

```python
loss = torch.tensor(3.5)
loss.shape      # torch.Size([])    a 0-dim tensor. Not a 1-element vector — no dims at all.
loss.item()     # 3.5               converts to a real Python float
```

Loss values are 0-dim tensors. `.item()` when you want to print or compare in Python.
(Tests check `out.shape == ()` for this, so it matters.)

Never compare floats with `==`:

```python
torch.allclose(a, b)                 # True/False, with sensible tolerance
torch.testing.assert_close(a, b)     # raises with a detailed diff — what the tests use
```

When a test fails, read its output properly: it reports the largest absolute and relative
difference. A mismatch of `1e-8` is floating-point noise and means you're right. A
mismatch of `0.5` means you're wrong.

---

## 10. Devices (one paragraph, then forget it)

```python
torch.backends.mps.is_available()    # True on your M5
x = torch.randn(3, 4, device="mps")  # lives on the GPU
x.to("cpu")                          # move it back
```

Tensors must be on the same device to interact. **Everything in Stage 00 and 01 runs on
CPU** — these tensors are tiny and the GPU would be slower. Devices first matter in
Stage 10. Don't think about it yet.

---

## The whole cheat sheet

Everything the seven drills need:

| Need | Call |
|------|------|
| make | `torch.zeros`, `torch.ones`, `torch.arange`, `torch.tensor`, `torch.tril` |
| inspect | `.shape`, `.dtype`, `.ndim` |
| convert | `.float()`, `.long()`, `.bool()`, `.item()` |
| index | `x[i]`, `x[:, i]`, `x[a:b]`, `x[rows, cols]`, `x[mask]` |
| maths | `+ - * / **`, `.exp()`, `.log()`, `@` |
| reduce | `.sum(dim=, keepdim=)`, `.mean(dim=)`, `.max(dim=).values` |
| reshape | `.view()`, `.reshape()`, `.transpose()`, `.contiguous()` |

That's it. That is the entire API surface of stage 00.

---

## Predict, then run

Do not skip this. Write your prediction down — shape *and* values — then type it in. The
ones you get wrong are precisely the things you'd otherwise discover at 1am inside a
broken attention implementation.

```python
x = torch.arange(6).reshape(2, 3).float()

# 1.  x.sum(dim=0).shape       and the values?
# 2.  x.sum(dim=1, keepdim=True).shape
# 3.  (x / x.sum(dim=1, keepdim=True))        what does each row now sum to?
# 4.  x / x.sum(dim=1)                        does this work? if not, why not?
# 5.  x[torch.tensor([1, 1, 0])]              shape?
# 6.  x[torch.tensor([0, 1]), torch.tensor([2, 0])]    shape, and which elements?
# 7.  x.view(3, 2)                            where does the number 3 end up?
# 8.  x.transpose(0, 1)                       where does the number 3 end up now?
# 9.  x.view(3, 2) == x.transpose(0, 1)       same shape — same contents?
# 10. torch.ones(2, 3) @ torch.ones(3, 2)     shape, and what's in every cell?
# 11. torch.tril(torch.ones(3, 3))            draw it before you run it
# 12. torch.tensor([1000., 1001.]).exp()      predict this one carefully
```

Question 9 is the important one. If you can explain *why* those two differ, you've
understood the trap in drill 6 before you meet it.

---

## What you're deliberately NOT learning yet

PyTorch tutorials front-load these. You don't need any of them, and skipping them is the
point — each one arrives later attached to a reason.

- `nn.Module`, `nn.Linear`, `nn.Embedding` — Stage 04, after you've built them yourself
- autograd, `requires_grad`, `.backward()` — Stage 01, after you've written your own
- optimizers, `DataLoader`, `Dataset`, training loops — Stages 03 and 09
- CUDA, mixed precision, `torch.compile` — Stage 11

---

## Looking things up

- **[pytorch.org/docs/stable/torch.html](https://pytorch.org/docs/stable/torch.html)** —
  the function index. Searching this is a core skill; use it constantly, it's not cheating.
- **[Tensors tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)**
  — 10 minutes, overlaps this primer. Worth it if you want a second phrasing.
- **In the REPL:** `help(torch.tril)` or `torch.tril?` in IPython beats a web search.
- **Know numpy?** Nearly everything transfers. The renames: `axis=` → `dim=`,
  `np.reshape` → `.view`/`.reshape`, `keepdims` → `keepdim`.

Now go do the drills. Expect to look things up — that's the normal condition of writing
PyTorch, not a sign you skipped something.
