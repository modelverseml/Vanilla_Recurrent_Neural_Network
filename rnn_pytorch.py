"""Standalone PyTorch RNN, built to mirror the from-scratch model.

`TorchRNN` takes the same constructor inputs as
`rnn_scratch_multi_layer.RNN` (features-first tensors X/Y, a `hidden_layers`
tuple, learning rate, iterations, task) and trains natively with PyTorch. It
does *not* share weights with the manual model — `predict` runs the trained
network forward on the given input and returns predictions in the same
(n_y, m, T_x) layout, so `utils.evaluate` / `generate` can compare it directly.

It uses one `nn.RNN` per layer (rather than a single multi-layer `nn.RNN`) so
layers may have different hidden sizes, matching the `hidden_layers` tuple.

Layout note: the manual model uses (n_x, m, T_x); PyTorch (batch_first) expects
(m, T_x, n_x). The wrapper transposes on the way in and out.
"""

import numpy as np
import torch
import torch.nn as nn


class _Net(nn.Module):
    """Stack of single-layer nn.RNNs followed by a per-timestep linear output."""

    def __init__(self, n_x, hidden_layers, n_y):
        super().__init__()
        self.rnns = nn.ModuleList()
        in_size = n_x
        for units in hidden_layers:
            self.rnns.append(
                nn.RNN(in_size, units, batch_first=True, nonlinearity="tanh"))
            in_size = units
        self.fc = nn.Linear(in_size, n_y)   # per-timestep output

    def forward(self, x):
        for rnn in self.rnns:
            x, _ = rnn(x)                    # (m, T_x, units)
        return self.fc(x)                   # logits (m, T_x, n_y)


class TorchRNN:
    """Stacked RNN built with PyTorch."""

    def __init__(self, X, Y, hidden_layers=(100,), learning_rate=0.01,
                 iterations=1000, task="classification"):
        self.X = X
        self.Y = Y
        self.hidden_layers = list(hidden_layers)
        self.learning_rate = learning_rate
        self.iterations = iterations
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        self.n_x = X.shape[0]
        self.n_y = Y.shape[0]
        self.model = None

    def train(self):
        Xk = torch.tensor(np.transpose(self.X, (1, 2, 0)), dtype=torch.float32)
        Yk = np.transpose(self.Y, (1, 2, 0))        # numpy (m, T_x, n_y)

        self.model = _Net(self.n_x, self.hidden_layers, self.n_y)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)

        if self.task == "classification":
            criterion = nn.CrossEntropyLoss()        # applies softmax internally
            # Targets as class indices per timestep, shape (m, T_x).
            targets = torch.tensor(np.argmax(Yk, axis=2), dtype=torch.long)
        else:
            criterion = nn.MSELoss()
            targets = torch.tensor(Yk, dtype=torch.float32)

        self.model.train()
        for i in range(self.iterations):
            optimizer.zero_grad()
            logits = self.model(Xk)                  # (m, T_x, n_y)
            if self.task == "classification":
                m, T_x, n_y = logits.shape
                loss = criterion(logits.reshape(m * T_x, n_y),
                                 targets.reshape(m * T_x))
            else:
                loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"[PyTorch] iter {i}, loss {loss.item():.4f}")
        return self

    def predict(self, X):
        """Run the trained network forward. Returns (n_y, m, T_x)."""
        self.model.eval()
        with torch.no_grad():
            Xk = torch.tensor(np.transpose(X, (1, 2, 0)), dtype=torch.float32)
            logits = self.model(Xk)                  # (m, T_x, n_y)
            # Match the manual model's predict: probabilities for classification.
            out = (torch.softmax(logits, dim=2) if self.task == "classification"
                   else logits)
        return np.transpose(out.numpy(), (2, 0, 1))  # (n_y, m, T_x)
