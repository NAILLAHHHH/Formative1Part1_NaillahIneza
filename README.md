# Formative 1 Part 1

What I built, following the guide in order.

`nn/module.py` is the base class. `Linear` is `x @ W + b`. Weights are Xavier uniform, bias starts at zero, and backward writes `dW` and `db` into the arrays from `__init__` (not new ones). ReLU, sigmoid and softmax are in `nn/activations`. The two cross-entropy losses are not modules. SGD does `param -= lr * grad` and then clears the gradients in place.

`main.py` trains the AND gate with one linear layer, sigmoid and binary cross-entropy. XOR would not work with a single layer, so I did not use it. `accuracy()` calls `train()` itself if nothing has been trained yet.

From this folder, environment `iml-formative1`:

```bash
pytest
ruff check nn/
ruff check main.py
python main.py
```
