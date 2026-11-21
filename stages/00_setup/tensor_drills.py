"""Stage 00 — tensor fluency drills.

Fill in every function below. Run `uv run pytest tests/test_00_setup.py -v` until green.

Notation used throughout this repo (memorise it, it never changes):
    B  batch size            — how many sequences we process at once
    T  time / block size     — how many tokens in each sequence
    C  channels / n_embd     — the size of each token's vector
    V  vocab size            — how many distinct tokens exist
    nh number of heads
    hs head size             — always C // nh
"""

import torch


def one_hot(idx: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Turn integer indices into one-hot row vectors.

    Args:
        idx:         (N,) int64 tensor, every value in [0, num_classes)
        num_classes: V

    Returns:
        (N, V) float32 tensor. Row i is all zeros except a 1.0 at column idx[i].

    Banned: torch.nn.functional.one_hot

    Hint: start from a zeros tensor of the right shape and use indexing to place the 1s.
    `torch.arange` is your friend — think about what `out[rows, cols] = 1.0` does when
    `rows` and `cols` are both tensors.

    Why you care: an embedding lookup `emb[idx]` is mathematically identical to
    `one_hot(idx) @ emb.weight` — a matrix multiply by a one-hot vector just selects a
    row. Embeddings aren't a special kind of layer; they're a Linear layer with a
    shortcut. Seeing that now will save you confusion in Stage 03.
    """
    raise NotImplementedError


def stable_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Softmax over the LAST dimension, numerically stable.

    Args:
        logits: (..., V) float tensor. Any number of leading dimensions.

    Returns:
        Same shape. Every slice along the last dim is non-negative and sums to 1.

    Banned: torch.softmax, torch.nn.functional.softmax, torch.logsumexp

    Must not overflow: this has to work for `logits = torch.tensor([1000., 1001.])`.
    Subtract the per-row max (keepdim!) before exponentiating.
    """
    raise NotImplementedError


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean negative log likelihood of the target classes.

    Args:
        logits:  (N, V) raw scores — NOT probabilities, no softmax applied yet
        targets: (N,) int64, the correct class index for each row

    Returns:
        scalar tensor — the mean loss over the N rows

    Banned: F.cross_entropy, F.nll_loss, F.log_softmax

    You may use your own stable_softmax. But think about this before you do: you're about
    to compute log(exp(x) / sum(exp(x))). Going through the exponential and back through
    the log loses precision, and if a probability underflows to exactly 0.0 then log(0)
    is -inf and your loss is nan.

    The better route is log-softmax computed directly:

        log p_i = x_i - max(x) - log( sum_j exp(x_j - max(x)) )

    Derive that from the softmax definition yourself — it's three lines of algebra, and
    it's the form every real implementation uses. Then just select the target entries.

    Sanity check to run in your head: for uniform logits over V classes this must return
    log(V). For V=65, that's 4.174.
    """
    raise NotImplementedError


def gather_target_probs(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Pick out the probability assigned to each row's target class.

    Args:
        probs:   (N, V) — each row sums to 1
        targets: (N,) int64

    Returns:
        (N,) — out[i] == probs[i, targets[i]]

    Banned: torch.gather, any Python loop

    Hint: `probs[a, b]` with two tensors does *paired* indexing, not a cross-product.
    So you need an `a` that counts 0, 1, 2, ..., N-1.

    This 'arange trick' is everywhere in ML code. Know it cold.
    """
    raise NotImplementedError


def causal_mask(T: int) -> torch.Tensor:
    """The lower-triangular boolean mask that makes a model autoregressive.

    Args:
        T: sequence length

    Returns:
        (T, T) bool tensor. out[i, j] is True iff j <= i — meaning
        "when producing the output at position i, position j is allowed to be seen."

    Hint: torch.tril, torch.ones.

    Why you care: this single matrix is what stops the model from cheating by reading the
    answer. Remove it and you have BERT. Keep it and you have GPT.
    """
    raise NotImplementedError


def split_heads(x: torch.Tensor, n_head: int) -> torch.Tensor:
    """Split the channel dimension into separate attention heads.

    Args:
        x:      (B, T, C)
        n_head: nh, must divide C evenly

    Returns:
        (B, nh, T, hs) where hs = C // nh.

        Head h must get channels [h*hs : (h+1)*hs] of the original vector. The token and
        batch structure must be preserved exactly — head h at batch b, time t must hold
        x[b, t, h*hs:(h+1)*hs].

    Hint: one `.view()` then one `.transpose()`. Think hard about the order.

    THE TRAP: `x.view(B, nh, T, hs)` also produces the right shape, and it is WRONG. It
    scrambles time and head together. The tests check values, not just shape, so they
    will catch it — but understand *why* it's wrong rather than permuting until green.
    Reason it through with the memory layout: `.view` reinterprets a contiguous block of
    numbers, and in `(B, T, C)` the numbers for a single token are adjacent.

    This is the single most common bug in from-scratch transformers. Get it right once,
    here, in isolation, where nothing else can be at fault.
    """
    raise NotImplementedError


def mean_of_past(x: torch.Tensor) -> torch.Tensor:
    """For each position t, average all positions 0..t inclusive.

    Args:
        x: (B, T, C)

    Returns:
        (B, T, C) where out[b, t] == x[b, 0:t+1].mean(dim=0)

    This is the ONE drill where you should use Python for-loops. Write the slow, obvious,
    obviously-correct version.

    Why you care: this is attention. Not an analogy for attention — in Stage 05 you'll
    see that it's exactly what you get when all the attention weights are equal. Every
    sophistication after that (learned weights, queries and keys, multiple heads) is a
    refinement of "each token looks at the tokens before it and mixes them together".

    You'll also rewrite this as a single matrix multiply in Stage 05 and watch it get
    ~100x faster. Keep this version around to check the fast one against.
    """
    raise NotImplementedError
