"""Standalone Keras (TensorFlow) RNN, built to mirror the from-scratch model.

`KerasRNN` takes the same constructor inputs as
`rnn_scratch_multi_layer.RNN` (the batch-first tensors X/Y, a `hidden_layers`
tuple, learning rate, epochs, batch_size, task) and trains natively with Keras. It does
*not* share weights with the manual model — `predict` simply runs the trained
Keras model forward on the given input and returns predictions in the same
(m, T_x, n_y) layout, so `utils.evaluate` / `generate` can compare it directly.

The batch-first layout (m, T_x, n_x) is exactly what Keras expects, so no
transposing is needed.
"""

import tensorflow as tf


class KerasRNN:
    """Stacked SimpleRNN (one per `hidden_layers` entry) with a per-timestep output."""

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
        # X/Y are already batch-first (m, T_x, n_x) / (m, T_x, n_y).
        self.model = self._build_model(self.X.shape[1])
        # Mini-batch training over epochs (Keras shuffles batches internally).
        self.model.fit(self.X, self.Y, epochs=self.epochs,
                       batch_size=self.batch_size, verbose=0)
        final_loss = self.model.evaluate(self.X, self.Y, verbose=0)
        print(f"[Keras] trained {self.epochs} epochs (batch_size={self.batch_size}), "
              f"final loss {final_loss:.4f}")
        return self

    def predict(self, X):
        """Run the trained Keras model forward. Returns (m, T_x, n_y)."""
        return self.model.predict(X, verbose=0)
