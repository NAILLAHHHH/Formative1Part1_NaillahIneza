"""Linear (fully connected) layer: z = xW + b."""

import numpy as np

from nn.module import Module


class Linear(Module):
    """A fully connected layer computing z = xW + b.

    Attributes:
        W (np.ndarray): weight matrix, shape (in_features, out_features).
        b (np.ndarray): bias vector, shape (out_features,).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize the layer's weights, bias, and gradient buffers.

        Args:
            in_features (int): number of input features.
            out_features (int): number of output neurons.

        Sets:
            self.W (np.ndarray): weight matrix, shape
                (in_features, out_features). Xavier uniform, not zeros.
            self.b (np.ndarray): bias vector, shape (out_features,). Zeros.
            self.dW (np.ndarray): gradient buffer, same shape as W, zeros.
                backward writes into this array in place.
            self.db (np.ndarray): gradient buffer, same shape as b, zeros.
        """
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute this layer's output for a batch of inputs.

        Args:
            x (np.ndarray): input, shape (batch_size, in_features).

        Returns:
            np.ndarray: output, shape (batch_size, out_features).

        Sets:
            self.x (np.ndarray): the input, kept so backward can use it.
        """
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Args:
            grad_output (np.ndarray): gradient of the loss with respect to this
                layer's output, shape (m, C).

        Returns:
            np.ndarray: gradient of the loss with respect to this layer's
            input, shape (m, n). Not summed over the batch.

        Sets:
            self.dW (np.ndarray): X.T @ grad_output, written in place.
            self.db (np.ndarray): grad_output summed over the batch, in place.
        """
        self.dW[...] = self.x.T @ grad_output
        self.db[...] = np.sum(grad_output, axis=0)
        return grad_output @ self.W.T

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return this layer's learnable parameters.

        Returns:
            list[tuple[np.ndarray, np.ndarray]]: (parameter, gradient) pairs,
            [(self.W, self.dW), (self.b, self.db)].
        """
        return [(self.W, self.dW), (self.b, self.db)]
