"""Standalone PyTorch RNN, built to mirror the from-scratch model.

`TorchRNN` takes the same constructor inputs as
`rnn_scratch_multi_layer.RNN` (batch-first tensors X/Y, a `hidden_layers`
tuple, learning rate, epochs, batch_size, task) and trains natively with PyTorch. It
does *not* share weights with the manual model — `predict` runs the trained
network forward on the given input and returns predictions in the same
(m, T_x, n_y) layout, so `utils.evaluate` / `generate` can compare it directly.

It uses one `nn.RNN` per layer (rather than a single multi-layer `nn.RNN`) so
layers may have different hidden sizes, matching the `hidden_layers` tuple.

The batch-first layout (m, T_x, n_x) is exactly what `nn.RNN(batch_first=True)`
expects, so no transposing is needed.
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
                 epochs=15, batch_size=32, task="classification"):
        self.X = X
        self.Y = Y
        self.hidden_layers = list(hidden_layers)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        self.n_x = X.shape[2]
        self.n_y = Y.shape[2]
        self.model = None

    def train(self):
        # X/Y are already batch-first (m, T_x, n_x) / (m, T_x, n_y).
        Xk = torch.tensor(self.X, dtype=torch.float32)

        self.model = _Net(self.n_x, self.hidden_layers, self.n_y)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)

        if self.task == "classification":
            criterion = nn.CrossEntropyLoss()        # applies softmax internally
            # Targets as class indices per timestep, shape (m, T_x).
            targets = torch.tensor(np.argmax(self.Y, axis=2), dtype=torch.long)
        else:
            criterion = nn.MSELoss()
            targets = torch.tensor(self.Y, dtype=torch.float32)

        m = Xk.shape[0]
        batch_size = self.batch_size or m                # None -> one full-batch step

        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(m)                      # reshuffle each epoch
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, m, batch_size):
                idx = perm[start:start + batch_size]
                xb, tb = Xk[idx], targets[idx]            # this mini-batch
                optimizer.zero_grad()
                logits = self.model(xb)                  # (b, T_x, n_y)
                if self.task == "classification":
                    b, T_x, n_y = logits.shape
                    loss = criterion(logits.reshape(b * T_x, n_y),
                                     tb.reshape(b * T_x))
                else:
                    loss = criterion(logits, tb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            print(f"[PyTorch] epoch {epoch + 1}/{self.epochs}, "
                  f"loss {epoch_loss / n_batches:.4f}")
        return self

    def predict(self, X):
        """Run the trained network forward. Returns (m, T_x, n_y)."""
        self.model.eval()
        with torch.no_grad():
            Xk = torch.tensor(X, dtype=torch.float32)
            logits = self.model(Xk)                  # (m, T_x, n_y)
            # Match the manual model's predict: probabilities for classification.
            out = (torch.softmax(logits, dim=2) if self.task == "classification"
                   else logits)
        return out.numpy()                           # (m, T_x, n_y)
