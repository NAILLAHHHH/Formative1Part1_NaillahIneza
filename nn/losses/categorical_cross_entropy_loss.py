"""Categorical cross-entropy loss, for one-hot multi-class targets."""

import numpy as np


class CategoricalCrossEntropyLoss:
    """Categorical cross-entropy loss over C classes.

    Does not subclass Module. Targets are one-hot, shape (m, C).
    """

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the average categorical cross-entropy loss.

        Args:
            predictions (np.ndarray): softmax probabilities, shape (m, C).
                Clipped away from exactly 0 before use.
            targets (np.ndarray): one-hot true labels, shape (m, C).

        Returns:
            float: the scalar loss, averaged over the batch.
        """
        self.a = np.clip(predictions, 1e-12, 1 - 1e-12)
        self.y = targets
        m = predictions.shape[0]
        return float(-np.sum(self.y * np.log(self.a)) / m)

    def backward(self) -> np.ndarray:
        """Compute the gradient of the loss with respect to predictions.

        Returns:
            np.ndarray: dL/da, shape (m, C). Uses the same clipped
            predictions as forward.
        """
        m = self.a.shape[0]
        return -(self.y / self.a) / m
