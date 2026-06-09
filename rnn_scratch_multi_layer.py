"""A multi-layer (stacked) vanilla RNN implemented from scratch with NumPy.

Trained with mini-batch backpropagation through time (BPTT) and plain SGD.
Supports softmax/cross-entropy classification and MSE regression.

Array conventions (NumPy, batch-first — the standard framework layout):
    X       : (m, T_x, n_x)   input sequence
    Y       : (m, T_x, n_y)   targets (one-hot for classification)
    hidden  : (m, T_x, n_a)   hidden state per layer
where m = batch size, T_x = sequence length, n_x = input features,
n_y = output dim, n_a = units in a given layer.

Because examples are stored as rows, weights left of the data become weights to
its right: a step is ``x @ Wax + a @ Waa + b`` (not ``Wax @ x``). Weight shapes
follow suit — ``Wax`` is (in, n_a), ``Waa`` is (n_a, n_a), ``Wya`` is (n_a, n_y).
"""

import numpy as np


class RNN:
    """Stacked Elman RNN with a single shared output projection over time."""

    def __init__(self, X, Y, hidden_layers=(100,), learning_rate=0.01, epochs=15,
                 batch_size=32, task="classification"):
        # X: (m, T_x, n_x) inputs, Y: (m, T_x, n_y) targets.
        self.X = X
        self.Y = Y
        self.hidden_layers = list(hidden_layers)        # units per stacked layer, bottom -> top
        self.L = len(self.hidden_layers)                # number of recurrent layers
        self.learning_rate = learning_rate
        self.epochs = epochs                            # passes over the training set
        self.batch_size = batch_size                    # sequences per mini-batch (None -> full batch)
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        self.n_x = X.shape[2]               # input feature dimension
        self.n_y = Y.shape[2]               # output dimension

        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        """Create weight matrices/biases for every layer plus the output layer."""
        # Per-layer parameter lists: index l holds the weights for hidden layer l.
        Wax, Waa, ba = [], [], []

        np.random.seed(1)

        for l in range(self.L):
            n_a = self.hidden_layers[l]                              # units in this layer
            # Layer 0 reads the raw input; deeper layers read the layer below's hidden state.
            in_size = self.n_x if l == 0 else self.hidden_layers[l-1]

            Wax.append(np.random.randn(in_size, n_a)*0.01)          # input -> hidden
            Waa.append(np.random.randn(n_a, n_a)*0.01)              # hidden -> hidden (recurrent)
            ba.append(np.zeros(n_a))                                # hidden bias

        # Output layer maps the top hidden layer's state to the output dimension n_y.
        Wya = np.random.randn(self.hidden_layers[-1], self.n_y)*0.01
        by = np.zeros(self.n_y)

        parameters = {
            'Wax' : Wax,
            'Waa' : Waa,
            'ba' : ba,
            'Wya' : Wya,
            'by' : by
        }
        return parameters

    def layer_forward(self, x_seq, a0, l):
        """Run recurrent layer l over the whole sequence.

        Returns the hidden-state sequence a (m, T_x, n_a) and a per-timestep
        cache used during backprop.
        """
        Wax = self.parameters['Wax'][l]
        Waa = self.parameters['Waa'][l]
        ba = self.parameters['ba'][l]

        m, T_x, _ = x_seq.shape           # batch size, sequence length, input feature dim
        n_a = self.hidden_layers[l]

        a_prev = a0                       # initial hidden state for this layer, (m, n_a)
        a = np.zeros((m, T_x, n_a))       # stores hidden state at every timestep
        layer_cache = []                  # per-timestep values needed for backprop

        for t in range(T_x):
            xt = x_seq[:, t, :]           # input at timestep t, (m, in)
            a_next = np.tanh(
                np.dot(xt, Wax) + np.dot(a_prev, Waa) + ba
            )
            a[:, t, :] = a_next
            layer_cache.append((a_prev, a_next, xt))
            a_prev = a_next               # carry hidden state to next timestep
        return a, layer_cache

    def rnn_forward(self, X, a0):
        """Full forward pass: stack the layers, then project to the output.

        a0 is a list of initial hidden states, one per layer. Returns
        (caches, a_top, y_pred).
        """
        # Forward pass through all stacked layers, then the output layer.
        caches_per_layer = []

        inp = X                                   # layer 0 reads the raw input sequence
        for l in range(self.L):
            # Each layer consumes the hidden-state sequence produced by the layer below.
            a, layer_caches = self.layer_forward(inp, a0[l], l)
            caches_per_layer.append(layer_caches)
            inp = a

        a_top = inp                               # hidden states of the topmost layer
        Wya = self.parameters["Wya"]
        by = self.parameters["by"]
        # Apply the output weights at every timestep: (m,T_x,n_a) x (n_a,n_y) -> (m,T_x,n_y).
        z = np.einsum("mtk,kj->mtj", a_top, Wya) + by
        y_pred = self.softmax(z) if self.task == "classification" else z

        caches = (caches_per_layer, X)
        return caches, a_top, y_pred

    def rnn_cell_forward(self, xt, a_prev):
        """One forward timestep through the whole stack (used for generation).

        Args:
            xt     : input at this step, shape (m, n_x).
            a_prev : list of L hidden states, one per layer, each (m, n_a_l).
        Returns:
            a_next : list of L updated hidden states (feed back in next step).
            y_pred : output distribution / values at this step, shape (m, n_y).
        """
        a_next = []
        inp = xt                                  # bottom layer reads the raw input
        for l in range(self.L):
            Wax = self.parameters['Wax'][l]
            Waa = self.parameters['Waa'][l]
            ba = self.parameters['ba'][l]
            a_l = np.tanh(np.dot(inp, Wax) + np.dot(a_prev[l], Waa) + ba)
            a_next.append(a_l)
            inp = a_l                             # output of this layer feeds the next
        # Project the top layer's state to the output.
        z = np.dot(inp, self.parameters["Wya"]) + self.parameters["by"]
        y_pred = self.softmax(z) if self.task == "classification" else z
        return a_next, y_pred

    def compute_loss(self, y_pred, Y):
        """Average loss over the batch (summed over all timesteps)."""
        # Cross-entropy for classification, mean squared error for regression.
        m = Y.shape[0]
        if self.task == "classification":
            y_pred = np.clip(y_pred, 1e-12, 1.0)          # avoid log(0)
            return -np.sum(Y * np.log(y_pred)) / m
        return 0.5 * np.sum((y_pred - Y) ** 2) / m

    def rnn_backward(self, da, caches):
        """Backprop the hidden-state gradient da down through all layers.

        da is the gradient w.r.t. the top layer's hidden states. Returns a
        dict of per-layer weight gradients.
        """
        # Backprop from the top layer down to layer 0.
        (caches_per_layer, X) = caches

        dWax = [None]*self.L
        dWaa = [None]*self.L
        dba = [None]*self.L
        da_above = da                          # gradient arriving from the layer above
        for l in reversed(range(self.L)):
            dWax_l, dWaa_l, dba_l, dx = self.layer_backward(
                da_above, caches_per_layer[l], l
            )
            dWax[l] = dWax_l
            dWaa[l] = dWaa_l
            dba[l] = dba_l
            da_above = dx                      # this layer's input grad feeds the layer below

        return {"dWax": dWax, "dWaa": dWaa, "dba": dba}

    def layer_backward(self, da_above, layer_cache, l):
        """BPTT for one layer. Returns its weight grads plus dx, the gradient
        w.r.t. its input sequence (which becomes da_above for the layer below).
        """
        # Backprop through time for a single recurrent layer l.
        Wax = self.parameters["Wax"][l]
        Waa = self.parameters["Waa"][l]

        m, T_x, n_a = da_above.shape
        in_size = Wax.shape[0]

        dWax_l = np.zeros_like(Wax)
        dWaa_l = np.zeros_like(Waa)
        dba_l = np.zeros(n_a)
        dx = np.zeros((m, T_x, in_size))      # grad w.r.t. this layer's input sequence
        da_next = np.zeros((m, n_a))          # grad flowing back from the next timestep

        for t in reversed(range(T_x)):

            (a_prev, a_next, xt) = layer_cache[t]

            # Total grad on a_next = from layer above (this timestep) + from next timestep.
            da_total = da_above[:, t, :] + da_next

            dtanh = (1 - a_next ** 2) * da_total              # backprop through tanh
            dWax_l += np.dot(xt.T, dtanh)                     # grad w.r.t. input weights
            dWaa_l += np.dot(a_prev.T, dtanh)                 # grad w.r.t. recurrent weights
            dba_l += np.sum(dtanh, axis=0)                    # grad w.r.t. hidden bias
            da_next = np.dot(dtanh, Waa.T)                    # grad passed to previous timestep

            dx[:, t, :] = np.dot(dtanh, Wax.T)                # grad passed to the layer below

        return dWax_l, dWaa_l, dba_l, dx

    def update_parameters(self, gradients):
        """Vanilla SGD step: param -= learning_rate * grad."""
        # Update each recurrent layer's weights and biases.
        for l in range(self.L):
            self.parameters["Wax"][l] -= self.learning_rate * gradients["dWax"][l]
            self.parameters["Waa"][l] -= self.learning_rate * gradients["dWaa"][l]
            self.parameters["ba"][l] -= self.learning_rate * gradients["dba"][l]

        # Update the shared output layer.
        self.parameters["Wya"] -= self.learning_rate * gradients["dWya"]
        self.parameters["by"] -= self.learning_rate * gradients["dby"]

    def train(self):
        """Mini-batch training: each epoch, shuffle and step over batches.

        For every mini-batch: forward pass, output-layer gradient, BPTT down the
        stack, gradient clipping, and one SGD update.
        """
        m = self.X.shape[0]
        batch_size = self.batch_size or m                # None -> one full-batch step

        for epoch in range(self.epochs):
            perm = np.random.permutation(m)              # reshuffle the sequences each epoch
            epoch_loss, n_batches = 0.0, 0

            for start in range(0, m, batch_size):
                idx = perm[start:start + batch_size]
                Xb, Yb = self.X[idx], self.Y[idx]        # this mini-batch

                # Zero initial hidden state for each layer (sized to this batch).
                a0 = [np.zeros((Xb.shape[0], n_a)) for n_a in self.hidden_layers]
                caches, a, y_pred = self.rnn_forward(Xb, a0)      # forward pass

                # For both softmax+cross-entropy and MSE, the output grad is (y_pred - Y).
                dy = y_pred - Yb                                  # (b, T_x, n_y)
                dWya = np.einsum("mtk,mtj->kj", a, dy)            # grad w.r.t. output weights
                dby = np.sum(dy, axis=(0, 1))                     # grad w.r.t. output bias

                # Propagate the output error into the top hidden layer, then backprop down.
                da = np.einsum("mtj,kj->mtk", dy, self.parameters["Wya"])
                gradients = self.rnn_backward(da, caches)
                gradients.update({'dWya': dWya, 'dby': dby})

                # Gradient clipping to [-5, 5] to curb exploding gradients in BPTT.
                for key in ("dWya", "dby"):
                    np.clip(gradients[key], -5, 5, out=gradients[key])
                for key in ("dWax", "dWaa", "dba"):
                    for arr in gradients[key]:                    # one array per layer
                        np.clip(arr, -5, 5, out=arr)

                self.update_parameters(gradients)                 # gradient descent step
                epoch_loss += self.compute_loss(y_pred, Yb)
                n_batches += 1

            print("Epoch {}/{}, Loss: {:.4f}".format(epoch + 1, self.epochs,
                                                      epoch_loss / n_batches))

    def predict(self, X):
        """Run a forward pass through the trained RNN and return predictions."""
        # One zero initial hidden state per layer (rnn_forward indexes a0 per layer).
        a0 = [np.zeros((X.shape[0], n_a)) for n_a in self.hidden_layers]
        _, _, y_pred = self.rnn_forward(X, a0)           # forward returns (caches, a_top, y_pred)
        return y_pred

    def softmax(self, x):
        """Numerically stable softmax over the last axis (features)."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)
