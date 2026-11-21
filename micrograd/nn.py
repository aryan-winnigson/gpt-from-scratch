"""A tiny neural network library built on the Value engine.

Three classes, each containing the next: Neuron -> Layer -> MLP. This mirrors exactly
how torch.nn is organised, and how mygpt/ will be organised later. Everything exposes
parameters() so the training loop can find the weights without knowing the structure.
"""

import random

from .engine import Value


class Module:
    """Base class. Gives every network a zero_grad() and a parameters() contract."""

    def zero_grad(self):
        """Reset every parameter's gradient to 0.

        This exists because gradients ACCUMULATE (the += in your backward closures).
        Without this call, step 2 trains on step 1's gradients plus its own, step 3 on
        all three, and your updates grow without bound.

        This is literally what torch's optimizer.zero_grad() does.
        """
        raise NotImplementedError

    def parameters(self):
        """Every tunable Value in this module. Default: none."""
        return []


class Neuron(Module):
    """One neuron: out = tanh(w · x + b)."""

    def __init__(self, n_in: int, nonlin: bool = True):
        """Create n_in weights and one bias.

        Initialise the weights to random values in roughly [-1, 1] — random.uniform is
        fine here. The bias conventionally starts at 0.

        Do NOT initialise the weights to zero. Try it once, later, and watch the network
        fail to learn: every neuron in a layer would compute the same thing, receive the
        same gradient, and update identically. They'd stay clones forever. Randomness is
        what breaks that symmetry. (Stage 08 replaces 'random enough' with a principled
        scheme — there's real theory about the right scale.)

        Args:
            n_in:   number of inputs
            nonlin: if True apply .tanh() to the output, if False return the raw sum.
                    The output layer of a regression net wants nonlin=False — think
                    about what a squashing function on the final layer would do if your
                    targets lived outside its range.

        Why tanh and not relu here? Try relu once you're green and watch this tiny
        network sometimes get permanently stuck at a loss of ~2.67. With only 4 neurons
        per layer, if the unlucky initial weights push them all into ReLU's flat region
        they receive zero gradient and never recover — the 'dying ReLU' problem, live.
        tanh has a non-zero slope everywhere, so it can always crawl back out. Real
        transformers use GELU for essentially this reason; you'll build it in Stage 04.
        """
        raise NotImplementedError

    def __call__(self, x):
        """Forward pass. x is a list of Values (or floats) of length n_in.

        Compute w·x + b, then apply tanh if self.nonlin.

        Hint: sum(...) with a start value, or build it with zip. Remember that __radd__
        is defined, so Python's builtin sum() works on Values.
        """
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def __repr__(self):
        kind = "Tanh" if getattr(self, "nonlin", False) else "Linear"
        return f"{kind}Neuron({len(self.w)})"


class Layer(Module):
    """A row of independent neurons, all seeing the same input."""

    def __init__(self, n_in: int, n_out: int, **kwargs):
        raise NotImplementedError

    def __call__(self, x):
        """Returns a list of n_out Values — or the bare Value if n_out == 1.

        That unwrapping is a convenience so a 1-output network returns a scalar instead
        of a 1-element list. Real libraries don't do this; micrograd does, and it makes
        the training loop read nicely.
        """
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """A stack of Layers. The classic multi-layer perceptron."""

    def __init__(self, n_in: int, n_outs: list[int]):
        """MLP(3, [4, 4, 1]) is: 3 inputs -> 4 -> 4 -> 1 output.

        Every layer is nonlin=True EXCEPT the last, which must be nonlin=False so the
        network can produce negative outputs.

        Hint: build the full size list `[n_in] + n_outs`, then pair consecutive entries.
        """
        raise NotImplementedError

    def __call__(self, x):
        """Feed x through every layer in turn."""
        raise NotImplementedError

    def parameters(self):
        """Flatten the parameters of every layer into one list.

        Later, in PyTorch, model.parameters() does exactly this — walks the module tree
        and yields every tensor registered as a Parameter.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
