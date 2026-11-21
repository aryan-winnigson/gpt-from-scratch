"""Stage 00 tests — every drill checked against PyTorch's own implementation.

Run:  uv run pytest tests/test_00_setup.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from _util import load_stage_module

drills = load_stage_module("00_setup", "tensor_drills")

torch.manual_seed(1337)


# ---------------------------------------------------------------- environment

def test_torch_and_mps_available():
    assert torch.__version__ >= "2.5"
    # Not a hard requirement to pass the stage, but you want this True on an M-series Mac.
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available — CPU training will be slow but everything works")


# ------------------------------------------------------------------ drill 1

def test_one_hot_matches_torch():
    idx = torch.randint(0, 10, (64,))
    got = drills.one_hot(idx, 10)
    want = F.one_hot(idx, 10).float()
    assert got.shape == want.shape, f"expected {want.shape}, got {got.shape}"
    assert got.dtype.is_floating_point, "return floats, not ints — you'll matmul with these"
    torch.testing.assert_close(got, want)


def test_one_hot_is_an_embedding_lookup():
    """The point of the drill: one_hot @ W is exactly a row lookup."""
    idx = torch.randint(0, 10, (7,))
    W = torch.randn(10, 4)
    torch.testing.assert_close(drills.one_hot(idx, 10) @ W, W[idx])


# ------------------------------------------------------------------ drill 2

@pytest.mark.parametrize("shape", [(5,), (4, 10), (2, 3, 7)])
def test_stable_softmax_matches_torch(shape):
    logits = torch.randn(*shape) * 5
    got = drills.stable_softmax(logits)
    torch.testing.assert_close(got, torch.softmax(logits, dim=-1))


def test_stable_softmax_sums_to_one():
    probs = drills.stable_softmax(torch.randn(4, 10) * 5)
    torch.testing.assert_close(probs.sum(-1), torch.ones(4))
    assert (probs >= 0).all()


def test_stable_softmax_survives_huge_logits():
    """Without the max-subtraction trick this returns nan."""
    logits = torch.tensor([1000.0, 1001.0, 1002.0])
    got = drills.stable_softmax(logits)
    assert torch.isfinite(got).all(), "overflowed — subtract the row max before exp()"
    torch.testing.assert_close(got, torch.softmax(logits, dim=-1))


# ------------------------------------------------------------------ drill 3

def test_cross_entropy_matches_torch():
    logits = torch.randn(32, 20) * 3
    targets = torch.randint(0, 20, (32,))
    got = drills.cross_entropy(logits, targets)
    want = F.cross_entropy(logits, targets)
    assert got.shape == (), f"should be a scalar, got shape {tuple(got.shape)}"
    torch.testing.assert_close(got, want)


def test_cross_entropy_uniform_baseline():
    """The sanity number you'll check on step 0 of every training run: loss == log(V)."""
    V = 65
    loss = drills.cross_entropy(torch.zeros(8, V), torch.randint(0, V, (8,)))
    torch.testing.assert_close(loss, torch.tensor(float(torch.log(torch.tensor(V * 1.0)))))


def test_cross_entropy_is_stable_at_extremes():
    logits = torch.tensor([[0.0, 500.0], [500.0, 0.0]])
    targets = torch.tensor([1, 0])
    loss = drills.cross_entropy(logits, targets)
    assert torch.isfinite(loss), "nan/inf — go through log-softmax, not softmax-then-log"
    torch.testing.assert_close(loss, F.cross_entropy(logits, targets))


# ------------------------------------------------------------------ drill 4

def test_gather_target_probs():
    probs = torch.softmax(torch.randn(16, 12), dim=-1)
    targets = torch.randint(0, 12, (16,))
    got = drills.gather_target_probs(probs, targets)
    want = probs[torch.arange(16), targets]
    assert got.shape == (16,), f"expected (16,), got {tuple(got.shape)}"
    torch.testing.assert_close(got, want)
    # The classic bug: indexing with a single tensor gives you (16, 12), not (16,).
    assert got.numel() == 16, "you built the cross-product, not the paired selection"


# ------------------------------------------------------------------ drill 5

def test_causal_mask():
    m = drills.causal_mask(5)
    assert m.shape == (5, 5)
    assert m.dtype == torch.bool, "return a bool mask"
    torch.testing.assert_close(m, torch.tril(torch.ones(5, 5, dtype=torch.bool)))
    assert m[0, 0] and not m[0, 1], "position 0 may see itself but not the future"
    assert m[4].all(), "the last position may see everything"


# ------------------------------------------------------------------ drill 6

def test_split_heads_shape_and_values():
    B, T, C, nh = 2, 5, 12, 3
    hs = C // nh
    x = torch.randn(B, T, C)
    got = drills.split_heads(x, nh)
    assert got.shape == (B, nh, T, hs), f"expected {(B, nh, T, hs)}, got {tuple(got.shape)}"

    # Value check: head h, time t must be the h-th slice of that token's vector.
    for h in range(nh):
        torch.testing.assert_close(got[:, h], x[:, :, h * hs:(h + 1) * hs])


def test_split_heads_catches_the_wrong_view():
    """x.view(B, nh, T, hs) has the right shape and the wrong contents. Prove you didn't."""
    B, T, C, nh = 2, 4, 8, 2
    x = torch.randn(B, T, C)
    wrong = x.view(B, nh, T, C // nh)
    got = drills.split_heads(x, nh)
    assert not torch.allclose(got, wrong), (
        "you used .view() straight to (B, nh, T, hs) — that interleaves time and heads. "
        "view to (B, T, nh, hs) first, THEN transpose."
    )


# ------------------------------------------------------------------ drill 7

def test_mean_of_past():
    B, T, C = 2, 6, 3
    x = torch.randn(B, T, C)
    got = drills.mean_of_past(x)
    assert got.shape == x.shape
    for b in range(B):
        for t in range(T):
            torch.testing.assert_close(got[b, t], x[b, :t + 1].mean(dim=0))


def test_mean_of_past_does_not_peek_at_the_future():
    """Change the last token. Every earlier output must be untouched."""
    x = torch.randn(1, 6, 3)
    before = drills.mean_of_past(x)
    x2 = x.clone()
    x2[0, -1] += 100.0
    after = drills.mean_of_past(x2)
    torch.testing.assert_close(before[:, :-1], after[:, :-1])
