"""Stage 01 tests — your autograd checked against PyTorch's.

Every gradient your engine computes is compared to the gradient torch computes for the
identical expression. If these pass, your implementation of backpropagation is correct.

Run:  uv run pytest tests/test_01_autograd.py -v
"""

import random

import pytest
import torch

from micrograd.engine import Value
from micrograd.nn import MLP, Layer, Neuron

TOL = 1e-6


def compare(build_mine, build_torch, inputs):
    """Build the same expression both ways, check forward value and every gradient.

    build_mine:  fn(list[Value])  -> Value
    build_torch: fn(list[Tensor]) -> Tensor
    """
    mine = [Value(v) for v in inputs]
    out_mine = build_mine(mine)
    out_mine.backward()

    theirs = [torch.tensor(v, dtype=torch.double, requires_grad=True) for v in inputs]
    out_theirs = build_torch(theirs)
    out_theirs.backward()

    assert abs(out_mine.data - out_theirs.item()) < TOL, (
        f"forward value wrong: yours {out_mine.data}, torch {out_theirs.item()}"
    )
    for i, (m, t) in enumerate(zip(mine, theirs)):
        assert abs(m.grad - t.grad.item()) < TOL, (
            f"grad of input {i} wrong: yours {m.grad}, torch {t.grad.item()}"
        )


# ------------------------------------------------------------------ primitives

def test_add_the_worked_example():
    compare(lambda v: v[0] + v[1], lambda t: t[0] + t[1], [2.0, -3.0])


def test_mul():
    compare(lambda v: v[0] * v[1], lambda t: t[0] * t[1], [2.0, -3.0])


def test_pow():
    compare(lambda v: v[0] ** 3, lambda t: t[0] ** 3, [2.0])
    compare(lambda v: v[0] ** -2, lambda t: t[0] ** -2, [4.0])
    compare(lambda v: v[0] ** 0.5, lambda t: t[0] ** 0.5, [9.0])


def test_pow_rejects_value_exponent():
    with pytest.raises(AssertionError):
        Value(2.0) ** Value(3.0)


def test_exp():
    compare(lambda v: v[0].exp(), lambda t: t[0].exp(), [1.5])


def test_log():
    compare(lambda v: v[0].log(), lambda t: t[0].log(), [3.0])


def test_tanh():
    compare(lambda v: v[0].tanh(), lambda t: torch.tanh(t[0]), [0.7])
    compare(lambda v: v[0].tanh(), lambda t: torch.tanh(t[0]), [-2.5])


def test_relu():
    compare(lambda v: v[0].relu(), lambda t: torch.relu(t[0]), [3.0])
    compare(lambda v: v[0].relu(), lambda t: torch.relu(t[0]), [-3.0])


# ------------------------------------------------------------------ derived ops (free if primitives are right)

def test_derived_ops():
    compare(lambda v: v[0] - v[1], lambda t: t[0] - t[1], [2.0, 5.0])
    compare(lambda v: v[0] / v[1], lambda t: t[0] / t[1], [6.0, 3.0])
    compare(lambda v: -v[0], lambda t: -t[0], [4.0])
    compare(lambda v: 10.0 - v[0], lambda t: 10.0 - t[0], [4.0])
    compare(lambda v: 10.0 / v[0], lambda t: 10.0 / t[0], [4.0])
    compare(lambda v: 2.0 + v[0] * 3.0, lambda t: 2.0 + t[0] * 3.0, [4.0])
    compare(lambda v: v[0].sigmoid(), lambda t: torch.sigmoid(t[0]), [0.4])


def test_builtin_sum_works():
    """Needs __radd__, because sum() starts from the integer 0."""
    compare(lambda v: sum(v), lambda t: t[0] + t[1] + t[2], [1.0, 2.0, 3.0])


# ------------------------------------------------------------------ the two classic bugs

def test_node_used_twice_accumulates():
    """x * x. If you wrote `=` instead of `+=`, this gives 3.0 instead of 6.0."""
    x = Value(3.0)
    y = x * x
    y.backward()
    assert abs(x.grad - 6.0) < TOL, (
        f"got {x.grad}, expected 6.0 — your backward overwrites instead of accumulating"
    )


def test_diamond_graph():
    """a feeds two branches that merge again. Gradient must sum over both paths."""
    compare(
        lambda v: (v[0] * 2.0 + v[0] * 3.0) * v[0],
        lambda t: (t[0] * 2.0 + t[0] * 3.0) * t[0],
        [2.0],
    )


def test_self_add():
    x = Value(5.0)
    y = x + x
    y.backward()
    assert abs(x.grad - 2.0) < TOL, f"got {x.grad}, expected 2.0"


def test_deep_chain_requires_topological_order():
    """40 chained ops. Any ordering mistake and the gradient won't reach the start."""
    def mine(v):
        x = v[0]
        for _ in range(40):
            x = (x * 1.1 + 0.1).tanh()
        return x

    def theirs(t):
        x = t[0]
        for _ in range(40):
            x = torch.tanh(x * 1.1 + 0.1)
        return x

    compare(mine, theirs, [0.5])


def test_wide_shared_subexpression():
    """One node consumed by many downstream nodes — tests accumulation at scale."""
    def mine(v):
        shared = (v[0] * v[1]).tanh()
        return sum(shared * float(k) for k in range(1, 8))

    def theirs(t):
        shared = torch.tanh(t[0] * t[1])
        return sum(shared * float(k) for k in range(1, 8))

    compare(mine, theirs, [1.3, -0.7])


# ------------------------------------------------------------------ a real expression

def test_karpathy_neuron():
    """The canonical micrograd example: a single tanh neuron with two inputs."""
    def mine(v):
        x1, x2, w1, w2, b = v
        return (x1 * w1 + x2 * w2 + b).tanh()

    def theirs(t):
        x1, x2, w1, w2, b = t
        return torch.tanh(x1 * w1 + x2 * w2 + b)

    compare(mine, theirs, [2.0, 0.0, -3.0, 1.0, 6.8813735870195432])


def test_messy_composite():
    def mine(v):
        a, b = v
        c = a + b
        d = a * b + b ** 3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + (b + a).relu()
        d = d + 3 * d + (b - a).relu()
        e = c - d
        f = e ** 2
        g = f / 2.0
        g = g + 10.0 / f
        return g

    def theirs(t):
        a, b = t
        c = a + b
        d = a * b + b ** 3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + (b + a).relu()
        d = d + 3 * d + (b - a).relu()
        e = c - d
        f = e ** 2
        g = f / 2.0
        g = g + 10.0 / f
        return g

    compare(mine, theirs, [-4.0, 2.0])


def test_cross_entropy_by_hand():
    """Preview of Stage 04: softmax + NLL built from your primitives, grads vs torch."""
    logits = [1.2, -0.5, 2.3, 0.1]
    target = 2

    def mine(v):
        exps = [x.exp() for x in v]
        total = sum(exps)
        return -(exps[target] / total).log()

    def theirs(t):
        return -torch.log(torch.exp(t[target]) / sum(torch.exp(x) for x in t))

    compare(mine, theirs, logits)


# ------------------------------------------------------------------ nn.py

def test_neuron():
    n = Neuron(3)
    assert len(n.parameters()) == 4, "3 weights + 1 bias"
    out = n([1.0, 2.0, 3.0])
    assert isinstance(out, Value)
    assert -1.0 <= out.data <= 1.0, "nonlin=True means tanh, whose range is (-1, 1)"


def test_neuron_linear_mode():
    n = Neuron(2, nonlin=False)
    for w in n.w:
        w.data = -1.0
    n.b.data = 0.0
    # tanh(-2) is about -0.964, so an unsquashed -2.0 proves the activation was skipped.
    assert abs(n([1.0, 1.0]).data - (-2.0)) < TOL, "nonlin=False must not squash"


def test_layer():
    layer = Layer(3, 4)
    out = layer([1.0, 2.0, 3.0])
    assert isinstance(out, list) and len(out) == 4
    assert len(layer.parameters()) == 4 * (3 + 1)


def test_layer_single_output_unwraps():
    assert isinstance(Layer(3, 1)([1.0, 2.0, 3.0]), Value)


def test_mlp_structure():
    model = MLP(3, [4, 4, 1])
    assert len(model.parameters()) == 41, "4*(3+1) + 4*(4+1) + 1*(4+1) = 41"
    assert isinstance(model([1.0, 2.0, 3.0]), Value)


def test_mlp_last_layer_is_linear():
    """If the output layer had ReLU it could never emit a negative number."""
    model = MLP(2, [4, 1])
    assert model.layers[-1].neurons[0].nonlin is False


def test_zero_grad():
    model = MLP(3, [4, 1])
    model([1.0, 2.0, 3.0]).backward()
    assert any(p.grad != 0 for p in model.parameters()), "backward set no gradients at all"
    model.zero_grad()
    assert all(p.grad == 0 for p in model.parameters())


def test_mlp_actually_learns():
    """The real test: gradient descent on your engine must drive the loss down."""
    random.seed(1337)
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    model = MLP(3, [4, 4, 1])

    def total_loss():
        return sum((model(x) - y) ** 2 for x, y in zip(xs, ys))

    first = total_loss().data
    for _ in range(300):
        loss = total_loss()
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= 0.03 * p.grad

    final = total_loss().data
    assert final < first, f"loss went UP: {first:.4f} -> {final:.4f}"
    assert final < 0.1, f"loss only reached {final:.4f}; expected well under 0.1"


def test_forgetting_zero_grad_breaks_training():
    """Demonstrates WHY zero_grad exists. Gradients must pile up without it."""
    model = MLP(3, [4, 1])
    x = [1.0, 2.0, 3.0]
    model(x).backward()
    once = [p.grad for p in model.parameters()]
    model(x).backward()  # no zero_grad in between
    twice = [p.grad for p in model.parameters()]
    assert any(abs(b - 2 * a) < TOL and a != 0 for a, b in zip(once, twice)), (
        "gradients should have doubled — they accumulate until you clear them"
    )
