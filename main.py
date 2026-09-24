"""Train a single linear layer on the AND gate."""

import numpy as np

from nn.activations import Sigmoid
from nn.layers import Linear
from nn.losses import CrossEntropyLoss
from nn.optim import SGD

layer = None
activation = None
X = None
y = None


def toy_data() -> tuple[np.ndarray, np.ndarray]:
    """Return the AND-gate data from the guide.

    Returns:
        tuple[np.ndarray, np.ndarray]: X of shape (4, 2) and y of shape (4, 1).
        The label is 1 only for input (1, 1).
    """
    inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels = np.array([[0.0], [0.0], [0.0], [1.0]])
    return inputs, labels


def train(epochs: int = 4000, lr: float = 1.0, seed: int = 0) -> list[float]:
    """Train logistic regression on the AND gate.

    Args:
        epochs (int): number of updates.
        lr (float): learning rate passed to SGD.
        seed (int): seed for the weight initialisation.

    Returns:
        list[float]: loss after each epoch.
    """
    global layer, activation, X, y

    np.random.seed(seed)
    X, y = toy_data()
    layer = Linear(2, 1)
    activation = Sigmoid()
    loss_fn = CrossEntropyLoss()
    opt = SGD(layer.parameters(), lr)

    history = []
    for _ in range(epochs):
        probs = activation.forward(layer.forward(X))
        history.append(loss_fn.forward(probs, y))
        layer.backward(activation.backward(loss_fn.backward()))
        opt.step()
        opt.zero_grad()
    return history


def accuracy(loss_history: list[float] = None) -> float:
    """Accuracy of the current model on the AND data.

    Args:
        loss_history (list[float]): not used for the score. If train() has not
            been called yet, this runs train() with the default settings first.

    Returns:
        float: fraction of the four points predicted correctly. Probability
        at least 0.5 is class 1.
    """
    if layer is None:
        train()
    probs = activation.forward(layer.forward(X))
    guessed = (probs >= 0.5).astype(float)
    return float(np.mean(guessed == y))


if __name__ == "__main__":
    losses = train()
    print("loss at start", round(losses[0], 4))
    print("loss at end", round(losses[-1], 4))
    print("accuracy", round(accuracy(), 4))
