"""Binary cross-entropy loss."""

import numpy as np


class CrossEntropyLoss:
    """Binary cross-entropy loss for a single output probability.

    Does not subclass Module. backward takes no argument because the loss is
    where the chain starts.
    """

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the average binary cross-entropy loss.

        Args:
            predictions (np.ndarray): predicted probabilities, shape (m,) or
                (m, 1). Clipped away from exactly 0 or 1 before use.
            targets (np.ndarray): true labels, same shape as predictions,
                values 0 or 1.

        Returns:
            float: the scalar loss, averaged over the batch.
        """
        self.a = np.clip(predictions, 1e-12, 1 - 1e-12)
        self.y = targets
        terms = self.y * np.log(self.a) + (1 - self.y) * np.log(1 - self.a)
        return float(-np.mean(terms))

    def backward(self) -> np.ndarray:
        """Compute the gradient of the loss with respect to predictions.

        Returns:
            np.ndarray: dL/da, same shape as the predictions passed to
            forward. Uses the same clipped predictions as forward.
        """
        a = self.a
        y = self.y
        m = a.shape[0]
        return -(y / a - (1 - y) / (1 - a)) / m
