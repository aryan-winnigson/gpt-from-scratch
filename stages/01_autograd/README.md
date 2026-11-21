# Stage 01 — Autograd From Scratch

> **Time:** 3–5 hours. Take all of it.
> **You write:** `micrograd/engine.py`, then `micrograd/nn.py`
> **You pass:** `uv run pytest tests/test_01_autograd.py -v`

---

## Why this is the most important stage in the repo

You can build a transformer without understanding backpropagation. People do it every
day — they call `loss.backward()` and it works. But then the loss goes to `nan`, or the
gradients vanish, or a layer silently never learns, and they have no idea why, because
the thing that failed is a black box.

After this stage, `loss.backward()` will never be a black box again. You will have
written it. In about 100 lines.

And here's the thing that surprises people: **PyTorch's autograd works exactly the way
yours will.** Same algorithm, same data structure, same accumulation rule. The real
version adds tensors, GPU kernels, and a decade of optimisation — but the idea you're
about to implement is the whole idea. Nothing is withheld from you at this scale.

---

## 1. What a derivative actually means here

Forget limits and $dx$ for a moment. For our purposes:

> $\frac{\partial L}{\partial w}$ is the answer to: **"if I nudge $w$ up by a tiny amount,
> how much does $L$ change, and in which direction?"**

If $\frac{\partial L}{\partial w} = 3$, pushing $w$ up by 0.001 pushes $L$ up by about
0.003. We want $L$ (the loss) to go *down*, so we move $w$ in the opposite direction:

$$w \leftarrow w - \text{lr} \cdot \frac{\partial L}{\partial w}$$

That's gradient descent, in full. Every bit of training in this repo is that one line,
applied to millions of $w$s. The entire difficulty is **computing all those derivatives
efficiently**, and that's what autograd is for.

### Why not just nudge each weight and see?

You could! Change $w$ by 0.0001, recompute the loss, see how much it moved. That's the
*numerical* gradient, and it genuinely works — the tests in this stage use it as an
independent check.

The problem is cost. One forward pass per parameter. GPT-2 has 124 million parameters,
so one training step would cost 124 million forward passes. Backpropagation gets you
**every** derivative in the cost of roughly **one** extra forward pass. That factor of
124 million is the only reason deep learning exists at all.

---

## 2. The chain rule is the entire algorithm

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

Read it as a sentence: *"to know how $x$ affects the final loss, take how $y$ affects the
loss, and multiply by how $x$ affects $y$."*

Now the key reframe. Consider one operation deep inside the network, say $y = a \cdot b$.
This operation does **not** know what the loss is. It doesn't know what comes after it.
But it knows two things:

1. **Its own local derivatives**: $\frac{\partial y}{\partial a} = b$ and
   $\frac{\partial y}{\partial b} = a$. Pure local knowledge.
2. **What it's told from downstream**: somebody hands it $\frac{\partial L}{\partial y}$.

Multiply them and it can hand $\frac{\partial L}{\partial a} = \frac{\partial L}{\partial y} \cdot b$
back to whoever produced $a$.

So backpropagation is a **message-passing algorithm**. Each node receives a gradient from
its output, multiplies by its own local derivative, and passes the result to its inputs.
No node needs global knowledge. That's why it scales to 124 million parameters, and it's
why your implementation can be 100 lines.

### Derive these before you write any code

Open `notes/01_autograd.md` and work out $\frac{\partial y}{\partial a}$ (and
$\frac{\partial y}{\partial b}$ where it applies) for each. Actually write them down —
the act of deriving is the learning, and you will need every one of these again in
Stage 04 when you hand-derive LayerNorm.

| Operation | Local derivative(s) |
|-----------|---------------------|
| $y = a + b$ | ? |
| $y = a \cdot b$ | ? |
| $y = a^n$ (n constant) | ? |
| $y = e^a$ | ? (there's a shortcut: express it in terms of $y$) |
| $y = \log a$ | ? |
| $y = \tanh a$ | ? (again, express in terms of $y$ — it's $1 - y^2$, now prove it) |
| $y = \text{relu}(a)$ | ? (and what happens exactly at $a = 0$?) |

> The `exp` and `tanh` shortcuts matter in practice: during the backward pass you already
> have `y` sitting in memory from the forward pass, so reusing it is free. Real frameworks
> cache exactly these values. This is also why a forward pass costs memory, not just time —
> and why "activation memory" is a thing you'll hear about when people run out of VRAM.

---

## 3. The two things that trip everyone up

### (a) Topological order

Consider `d = a * b; e = d + c; L = e * 2`. To compute $\frac{\partial L}{\partial a}$ you
need $\frac{\partial L}{\partial d}$ first, which needs $\frac{\partial L}{\partial e}$
first. **You must process nodes in reverse order of computation.**

"Reverse order of computation" is precisely a **reverse topological sort** of the graph:
every node is visited only after all nodes that depend on it. You'll implement it with a
depth-first search that appends each node *after* recursing into its children, then
reverses the list.

Get this wrong and you don't crash — you just get wrong gradients, because some nodes
pass on a `grad` that's still 0 because their turn hadn't come. Silent, subtle, and the
reason the tests check gradients numerically.

### (b) Accumulation: `+=`, never `=`

```python
x = Value(3.0)
y = x * x        # x is used TWICE
```

$y = x^2$, so $\frac{\partial y}{\partial x} = 2x = 6$. But the multiply node will send a
gradient to its left input (value 3) *and* to its right input (value 3) — and they're the
same object. Each message is 3. If you write `self.grad = ...` the second message
overwrites the first and you get 3. If you write `self.grad += ...` you get 6. Correct.

The maths behind this is the **multivariable chain rule**: when a variable influences the
output through several paths, the total derivative is the *sum* over paths. A node used in
$k$ places receives $k$ messages, and they all add up.

This is also the reason `optimizer.zero_grad()` exists in PyTorch. Gradients accumulate by
design, so if you don't explicitly clear them between steps, step 2's gradients get added
on top of step 1's. Every person who has ever trained a model has forgotten this once and
watched their loss do something insane. Now you know exactly what the mechanism is.

---

## 4. Build it

### Part A — `micrograd/engine.py`

One class, `Value`, wrapping a single float. Four fields do all the work:

```python
self.data       # the number, computed on the forward pass
self.grad       # dL/d(this), filled in on the backward pass. Starts at 0.
self._prev      # the Values this one was computed from  → the graph edges
self._backward  # a closure: "push my grad to my inputs"  → the local derivative
```

The pattern for every operation is identical. Here is `__add__`, given to you in the file
as the worked example — study it until the shape of it is obvious, then write the other
six yourself:

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad  += 1.0 * out.grad     # d(out)/d(self)  is 1
        other.grad += 1.0 * out.grad     # d(out)/d(other) is 1
    out._backward = _backward

    return out
```

Three moves, every time: compute the forward value, record the parents, and define a
closure that distributes `out.grad` to the parents using the local derivative. That
closure captures `self`, `other`, and `out` from the enclosing scope — which is how it
still has access to the forward values later, when `backward()` runs.

Then `backward()` itself: topologically sort, seed `self.grad = 1.0` (because
$\frac{\partial L}{\partial L} = 1$), and call each node's `_backward()` in reverse order.

### Part B — `micrograd/nn.py`

Now use it. `Neuron` → `Layer` → `MLP`, each with a `parameters()` method. Then the
training loop, which is the same five lines you'll write in every stage from here on:

```python
for step in range(100):
    ypred = [model(x) for x in xs]                       # 1. forward
    loss = sum((yp - y)**2 for yp, y in zip(ypred, ys))  # 2. loss
    model.zero_grad()                                    # 3. clear old grads  ← don't skip
    loss.backward()                                      # 4. backward
    for p in model.parameters():                         # 5. step
        p.data -= 0.05 * p.grad
```

When you watch that loss go down, driven by derivatives you computed with code you wrote,
you'll have the thing most people never get.

### Experiment after the tests pass

Try these — they take two minutes each and each one teaches something you'd otherwise
learn the hard way at 2am in Stage 10:

1. Delete the `zero_grad()` call. What happens to the loss, and why *that* pattern?
2. Change `+=` to `=` in your `__mul__` backward. Which tests fail? Does the MLP still
   sort-of train?
3. Set the learning rate to 10. Then to 0.0001. Name what breaks in each case.
4. Initialise all weights to 0 instead of random. The network will refuse to learn.
   Work out why — the answer is called the *symmetry problem* and it's why nobody
   initialises to zero.
5. Swap `.tanh()` for `.relu()` in `Neuron.__call__` and re-run `test_mlp_actually_learns`
   a few times with different seeds. On some seeds it converges fine; on others it
   wedges at a loss of about 2.67 and never moves. Diagnose it before reading the note
   in `nn.py` — print the activations and find the neurons that output 0 for every
   input. That's the **dying ReLU** problem, and you just reproduced it on purpose.

---

## ✅ Before you move to Stage 02

1. Why is backpropagation faster than nudging each parameter and re-measuring? Give the
   rough cost of each for a model with $N$ parameters.
2. What are the four fields on a `Value` and what is each for?
3. Why must `backward()` walk nodes in reverse topological order? What goes wrong if you
   use insertion order instead — crash or silence?
4. Why `+=` and not `=`? Give the two-line code example where it matters.
5. What does `optimizer.zero_grad()` do in PyTorch, and what in your implementation is the
   exact thing it corresponds to?
6. For `y = tanh(x)`, why can the backward pass use `1 - y**2` instead of recomputing
   anything from `x`? What does that imply about memory during training?
7. Walk the graph of `L = (a*b + c).tanh()` by hand with `a=2, b=-3, c=10`. Forward values,
   then every gradient. Check yourself against your own code.

Then:

```bash
uv run pytest tests/test_01_autograd.py -v
# write notes/01_autograd.md — include your derivative table from section 2
git add -A && git commit -m "stage 01: scalar autograd engine + MLP"
git tag stage-01 && git push && git push --tags
```
