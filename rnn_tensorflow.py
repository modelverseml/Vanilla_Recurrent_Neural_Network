"""Standalone Keras (TensorFlow) RNN, built to mirror the from-scratch model.

`KerasRNN` takes the same constructor inputs as
`rnn_scratch_multi_layer.RNN` (the features-first tensors X/Y, a `hidden_layers`
tuple, learning rate, iterations, task) and trains natively with Keras. It does
*not* share weights with the manual model — `predict` simply runs the trained
Keras model forward on the given input and returns predictions in the same
(n_y, m, T_x) layout, so `utils.evaluate` / `generate` can compare it directly.

Layout note: the manual model uses (n_x, m, T_x); Keras expects
(m, T_x, n_x). The wrapper transposes on the way in and out.
"""

import numpy as np
import tensorflow as tf


class KerasRNN:
    """Stacked SimpleRNN (one per `hidden_layers` entry) with a per-timestep output."""

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

    def _build_model(self, T_x):
        keras = tf.keras
        model = keras.Sequential()
        model.add(keras.Input(shape=(T_x, self.n_x)))
        # return_sequences=True keeps an output at every timestep
        # (sequence-to-sequence, like the manual model).
        for units in self.hidden_layers:
            model.add(keras.layers.SimpleRNN(units, activation="tanh",
                                             return_sequences=True))
        out_act = "softmax" if self.task == "classification" else "linear"
        model.add(keras.layers.Dense(self.n_y, activation=out_act))

        loss = ("categorical_crossentropy" if self.task == "classification"
                else "mse")
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate),
                      loss=loss)
        return model

    def train(self):
        Xk = np.transpose(self.X, (1, 2, 0))    # (m, T_x, n_x)
        Yk = np.transpose(self.Y, (1, 2, 0))    # (m, T_x, n_y)

        self.model = self._build_model(self.X.shape[2])
        # Full-batch training so "iterations" matches the manual model's epochs.
        self.model.fit(Xk, Yk, epochs=self.iterations, batch_size=Xk.shape[0],
                       verbose=0)
        final_loss = self.model.evaluate(Xk, Yk, verbose=0)
        print(f"[Keras] trained {self.iterations} epochs, final loss {final_loss:.4f}")
        return self

    def predict(self, X):
        """Run the trained Keras model forward. Returns (n_y, m, T_x)."""
        Xk = np.transpose(X, (1, 2, 0))                 # (m, T_x, n_x)
        preds = self.model.predict(Xk, verbose=0)       # (m, T_x, n_y)
        return np.transpose(preds, (2, 0, 1))           # (n_y, m, T_x)
