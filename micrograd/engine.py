"""A scalar-valued automatic differentiation engine.

Every number in a computation becomes a Value node. Operations between Values build a
directed acyclic graph, and calling .backward() on the final node walks that graph in
reverse, filling in .grad on every node it depends on.

This is the same algorithm PyTorch uses. It is smaller because it handles one number at
a time instead of whole tensors, and because it has no GPU kernels. The idea is identical.

You implement: __mul__, __pow__, exp, log, tanh, relu, backward.
Given to you:  __add__ (the worked example), and the ops derived from the primitives.
"""


class Value:
    """A single scalar and its gradient, plus the edge back to whatever produced it."""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)

        # dL/d(self), where L is whatever node you eventually call .backward() on.
        # Starts at zero: "so far as we know, this number does not affect the loss."
        self.grad = 0.0

        # How to push self.grad back to the nodes that produced self.
        # A leaf (a weight, an input) has no parents, so its backward does nothing.
        self._backward = lambda: None

        # The parents. These are the edges of the graph.
        self._prev = set(_children)

        # Which operation produced this node. Debugging / visualisation only.
        self._op = _op

    # ------------------------------------------------------------------ primitives
    # Each one: compute the forward value, record the parents, and define a closure
    # that distributes out.grad to the parents using this op's local derivative.

    def __add__(self, other):
        """THE WORKED EXAMPLE. Read this until the pattern is obvious, then do the rest.

        out = self + other
        d(out)/d(self)  = 1
        d(out)/d(other) = 1
        An addition node is a pure 'gradient router' — it passes its gradient through
        to both inputs completely unchanged.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # += not =  — see stage README section 3(b). This is not a style choice.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        """out = self * other

        Work out d(out)/d(self) and d(out)/d(other) first, then write the closure.
        Note the pleasing symmetry: each input's gradient is scaled by the OTHER input's
        value. A multiply node is a 'gradient swapper'.

        Careful: this must still be correct when self is other (i.e. `x * x`).
        Think about what += buys you there.
        """
        raise NotImplementedError

    def __pow__(self, other):
        """out = self ** other, where other is a plain int or float (not a Value).

        Supporting a Value exponent means handling d/dn of a^n, which needs log and
        isn't worth it here — real frameworks do support it, we won't.

        Standard power rule. Watch the case self.data < 0 with a fractional exponent;
        you may assume the tests don't do that.
        """
        assert isinstance(other, (int, float)), "only int/float powers are supported"
        raise NotImplementedError

    def exp(self):
        """out = e ** self

        d(out)/d(self) = e ** self ... which is exactly out.data, already computed.
        Use that. Reusing the forward value is what real autograd engines do, and it's
        why training uses more memory than inference.
        """
        raise NotImplementedError

    def log(self):
        """out = ln(self). Natural log.

        You'll need this for cross-entropy. Assumes self.data > 0.
        """
        raise NotImplementedError

    def tanh(self):
        """out = tanh(self)

        tanh(x) = (e^2x - 1) / (e^2x + 1), and d/dx tanh(x) = 1 - tanh(x)^2.
        Prove that identity in your notes, then use the cheap form: 1 - out.data**2.

        This was THE activation function before ReLU. GPT-2 still uses a tanh-based
        approximation of GELU — you'll meet it again in Stage 04.
        """
        raise NotImplementedError

    def relu(self):
        """out = max(0, self)

        The derivative is 1 where self.data > 0 and 0 where self.data < 0. At exactly 0
        it's undefined; everyone picks 0 and nothing bad happens. (Why does nothing bad
        happen? Worth a minute's thought.)

        Note what the 0 branch means: a ReLU unit sitting in its flat region passes NO
        gradient to its inputs at all. If it gets stuck there it stops learning forever.
        That's the 'dying ReLU' problem, and it's one of several reasons transformers
        use GELU instead.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ the backward pass

    def backward(self):
        """Fill in .grad on every node this one depends on.

        Two steps:

        1. Build a topological ordering of the graph reachable from self — an order
           where every node appears after all the nodes it was built from. A DFS that
           appends a node AFTER recursing into node._prev gives you exactly this.

        2. Seed self.grad = 1.0 (because dL/dL = 1, and self IS the loss here), then
           call node._backward() on each node in REVERSE topological order.

        Why reverse: a node can only push gradient to its parents once its own .grad is
        final, which means every node downstream of it must have run already.

        Watch out for: visiting a node twice (use a `visited` set — the graph is a DAG,
        not a tree, and diamonds are common).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ derived ops
    # Already written — all of these are expressed in terms of the primitives above, so
    # they get correct gradients for free the moment your primitives are correct.
    # Read them: each one is a small lesson in composing operations.

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __truediv__(self, other):  # self / other
        return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)

    def __radd__(self, other):  # other + self   (lets you write `sum(list_of_values)`)
        return self + other

    def __rmul__(self, other):  # other * self
        return self * other

    def __rsub__(self, other):  # other - self
        return (-self) + other

    def __rtruediv__(self, other):  # other / self
        return (self ** -1) * other

    def sigmoid(self):  # 1 / (1 + e^-x)
        return 1.0 / (1.0 + (-self).exp())

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
