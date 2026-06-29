"""
A vanilla (Elman) RNN written from scratch in NumPy -- no deep-learning library.

Same idea as the PyTorch/TensorFlow models but with everything spelled out: one
recurrent layer, take the hidden state at the last real word, then a linear layer
into the class scores. Trained with full backprop-through-time (BPTT) and Adam,
with gradient clipping and best-dev checkpointing.

The training script (model_artifacts_generation.py) and the serving backend
(backend/predictor.py) both import ManualRNN from here.
"""

import numpy as np

# weight-init seed (matches SEED in model_artifacts_generation.py). kept local so
# this module has no imports from the training script (avoids a circular import).
DEFAULT_SEED = 42


class ManualRNN:
    """Vanilla RNN in pure NumPy.

    Shapes use a batch-first, row-vector convention:

        X : (batch, seq_len, input_dim)
        Wxh (input_dim, hidden)  Whh (hidden, hidden)  Why (hidden, classes)
    """

    def __init__(self, input_dim=None, hidden_dim=None, num_classes=None,
                 seed=DEFAULT_SEED):
        if input_dim is None:
            return  # empty shell, to be filled by load()
        rng = np.random.default_rng(seed)
        # tanh-friendly init: scale by 1/sqrt(fan_in)
        self.params = {
            "Wxh": rng.standard_normal((input_dim, hidden_dim)) / np.sqrt(input_dim),
            "Whh": rng.standard_normal((hidden_dim, hidden_dim)) / np.sqrt(hidden_dim),
            "Why": rng.standard_normal((hidden_dim, num_classes)) / np.sqrt(hidden_dim),
            "bh": np.zeros((1, hidden_dim)),
            "by": np.zeros((1, num_classes)),
        }
        self._init_adam()

    # ---- shapes / state ----
    @property
    def hidden_dim(self):
        return self.params["Whh"].shape[0]

    def _init_adam(self):
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

    # ---- forward / backward ----
    @staticmethod
    def _last_real_idx(X):
        # titles are zero-padded at the end; the last real word is the last row
        # that isn't all zeros (at least index 0 so empty titles don't break).
        mask = np.abs(X).sum(axis=-1) > 0          # (batch, seq_len)
        lengths = np.clip(mask.sum(axis=1), 1, None)
        return lengths - 1                         # (batch,)

    def forward(self, X):
        B, T, _ = X.shape
        H = self.hidden_dim
        Wxh, Whh, bh = self.params["Wxh"], self.params["Whh"], self.params["bh"]

        h_seq = np.zeros((B, T, H))
        h_prev = np.zeros((B, H))
        for t in range(T):
            a = X[:, t, :] @ Wxh + h_prev @ Whh + bh
            h_prev = np.tanh(a)
            h_seq[:, t, :] = h_prev

        last_idx = self._last_real_idx(X)
        h_last = h_seq[np.arange(B), last_idx]      # (B, H)
        logits = h_last @ self.params["Why"] + self.params["by"]
        cache = (X, h_seq, last_idx, h_last)
        return logits, cache

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def backward(self, logits, y, cache):
        X, h_seq, last_idx, h_last = cache
        B, T, _ = X.shape
        Whh, Why = self.params["Whh"], self.params["Why"]

        probs = self._softmax(logits)
        # cross-entropy gradient wrt logits, averaged over the batch
        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= B

        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        grads["Why"] = h_last.T @ dlogits
        grads["by"] = dlogits.sum(axis=0, keepdims=True)

        # gradient that flows into the hidden state at each example's last word
        dh_last = dlogits @ Why.T                   # (B, H)
        dh_next = np.zeros_like(dh_last)
        for t in reversed(range(T)):
            dh = dh_next.copy()
            sel = last_idx == t                     # inject output grad at last word
            dh[sel] += dh_last[sel]
            da = dh * (1.0 - h_seq[:, t, :] ** 2)   # tanh'
            grads["bh"] += da.sum(axis=0, keepdims=True)
            grads["Wxh"] += X[:, t, :].T @ da
            h_prev = h_seq[:, t - 1, :] if t > 0 else np.zeros_like(h_seq[:, 0, :])
            grads["Whh"] += h_prev.T @ da
            dh_next = da @ Whh.T
        return grads

    def _clip(self, grads, max_norm=5.0):
        total = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
        if total > max_norm:
            scale = max_norm / (total + 1e-6)
            for k in grads:
                grads[k] *= scale

    def _adam_step(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k in self.params:
            self._m[k] = b1 * self._m[k] + (1 - b1) * grads[k]
            self._v[k] = b2 * self._v[k] + (1 - b2) * (grads[k] ** 2)
            mhat = self._m[k] / (1 - b1 ** self._t)
            vhat = self._v[k] / (1 - b2 ** self._t)
            self.params[k] -= lr * mhat / (np.sqrt(vhat) + eps)

    # ---- public API ----
    def predict_proba(self, X):
        logits, _ = self.forward(X)
        return self._softmax(logits)

    def accuracy(self, X, y):
        preds = self.predict_proba(X).argmax(axis=1)
        return float((preds == y).mean())

    def fit(self, X_train, y_train, X_dev, y_dev, epochs, batch_size, lr):
        self._init_adam()
        n = len(X_train)
        best_dev, best_params = -1.0, None
        for epoch in range(epochs):
            order = np.random.permutation(n)        # seeded shuffle (set_seed)
            for s in range(0, n, batch_size):
                idx = order[s:s + batch_size]
                xb, yb = X_train[idx], y_train[idx]
                logits, cache = self.forward(xb)
                grads = self.backward(logits, yb, cache)
                self._clip(grads)
                self._adam_step(grads, lr)
            dev_acc = self.accuracy(X_dev, y_dev)
            if dev_acc > best_dev:
                best_dev = dev_acc
                best_params = {k: v.copy() for k, v in self.params.items()}
            print(f"  [manual] epoch {epoch + 1}/{epochs}  dev_acc={dev_acc:.3f}")
        if best_params is not None:
            self.params = best_params
        print(f"  [manual] best dev_acc={best_dev:.3f}")

    def save(self, path):
        np.savez(path, **self.params)

    @classmethod
    def load(cls, path):
        model = cls()                               # empty shell
        npz = np.load(path)
        model.params = {k: npz[k] for k in ("Wxh", "Whh", "Why", "bh", "by")}
        return model
