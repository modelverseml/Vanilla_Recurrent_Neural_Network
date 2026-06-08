import numpy as np


class RNN:
    """Recurrent Neural Network implemented from scratch in NumPy.

    The network maps an input sequence X to an output sequence Y one time step
    at a time, carrying a hidden state `a` forward through time. It is trained
    with full backpropagation through time (BPTT) and plain gradient descent.

    Shapes (used throughout):
        n_x : size of the input vector at each time step (e.g. vocabulary size)
        n_y : size of the output vector at each time step (e.g. vocabulary size)
        n_a : number of hidden units in the recurrent state
        m   : number of training examples (batch size)
        T_x : number of time steps in the sequence
    """

    def __init__(self, X, Y, n_a=100, learning_rate=0.01, iterations=1000,
                 task="classification"):
        # Training inputs/targets, expected shape (n_x, m, T_x) and (n_y, m, T_x).
        self.X = X
        self.Y = Y
        self.n_a = n_a                       # hidden state size
        self.learning_rate = learning_rate   # step size for gradient descent
        self.iterations = iterations         # number of training epochs
        # "classification" -> softmax output + cross-entropy loss (one-hot targets,
        #                     e.g. the character model).
        # "regression"     -> linear output + mean-squared-error loss (real-valued
        #                     targets, e.g. predicting word2vec vectors).
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        self.n_x = X.shape[0]                # input feature size
        self.n_y = Y.shape[0]                # output feature size

        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        """Create the weight matrices and bias vectors.

        Weights are initialized small (scaled by 0.01) to keep the initial
        activations in the linear region of tanh; biases start at zero.
        """
        np.random.seed(1)  # fixed seed so runs are reproducible
        Wax = np.random.randn(self.n_a, self.n_x) * 0.01  # input  -> hidden
        Waa = np.random.randn(self.n_a, self.n_a) * 0.01  # hidden -> hidden (recurrent)
        Wya = np.random.randn(self.n_y, self.n_a) * 0.01  # hidden -> output
        ba = np.zeros((self.n_a, 1))   # hidden bias
        by = np.zeros((self.n_y, 1))   # output bias

        parameters = {"Wax": Wax,
                      "Waa": Waa,
                      "Wya": Wya,
                      "ba": ba,
                      "by": by}

        return parameters

    def rnn_cell_forward(self, xt, a_prev):
        """Run one forward time step of the RNN.

        Args:
            xt     : input at this step,        shape (n_x, m)
            a_prev : hidden state from prev step, shape (n_a, m)
        Returns:
            a_next  : new hidden state,  shape (n_a, m)
            yt_pred : output prediction,  shape (n_y, m)
        """
        Wax = self.parameters["Wax"]
        Waa = self.parameters["Waa"]
        Wya = self.parameters["Wya"]
        ba = self.parameters["ba"]
        by = self.parameters["by"]

        # New hidden state: combine the current input with the previous state.
        a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
        # Output: project the hidden state. For classification, squash it into a
        # probability vector with softmax; for regression, leave it linear so the
        # network can output arbitrary real-valued targets (e.g. word vectors).
        z = np.dot(Wya, a_next) + by
        yt_pred = self.softmax(z) if self.task == "classification" else z

        cache = (a_prev, a_next, xt)

        return cache, a_next, yt_pred

    def rnn_forward(self, X, a0):
        """Run the forward pass over the whole sequence.

        Args:
            X : input sequence, shape (n_x, m, T_x)
        Returns:
            a      : every hidden state,  shape (n_a, m, T_x)
            y_pred : every prediction,    shape (n_y, m, T_x)
        """
        Wya = self.parameters["Wya"]

        caches = []

        n_x, m, T_x = X.shape
        n_y, n_a = Wya.shape

        a_next = a0                             # initial hidden state a<0> = 0
        a = np.zeros((n_a, m, T_x))            # store hidden states for backprop
        y_pred = np.zeros((n_y, m, T_x))       # store predictions

        # Step through time, feeding each step's hidden state into the next.
        for t in range(T_x):
            xt = X[:, :, t]
            cache, a_next, yt_pred = self.rnn_cell_forward(xt, a_next)
            a[:, :, t] = a_next
            y_pred[:, :, t] = yt_pred
            caches.append(cache)

        caches = (caches, X)
        return caches, a, y_pred

    def compute_loss(self, y_pred, Y):
        """Loss between predictions and targets, matched to the task.

        classification -> average cross-entropy (expects one-hot targets).
        regression      -> average mean-squared error (expects real-valued targets).

        In both cases the per-step output gradient is `y_pred - Y` (see
        rnn_backward), so the backward pass is shared between the two tasks.
        """
        m = Y.shape[1]
        if self.task == "classification":
            # Clip to avoid log(0) -> -inf / nan when a probability underflows.
            y_pred = np.clip(y_pred, 1e-12, 1.0)
            return -np.sum(Y * np.log(y_pred)) / m
        # regression: mean squared error (the 1/2 makes the gradient exactly y_pred - Y)
        return 0.5 * np.sum((y_pred - Y) ** 2) / m

    def rnn_cell_backward(self, da_next, cache):
        """Backprop through one time step.

        Args:
            da_next : total hidden-state gradient at this step. It is the gradient
                      coming from this step's output plus the gradient flowing back
                      from the *next* time step, shape (n_a, m).
            cache   : (a_prev, a_next, xt) saved during the forward pass:
                        a_prev : hidden state from the previous step, (n_a, m)
                        a_next : hidden state at this step,            (n_a, m)
                        xt     : input at this step,                   (n_x, m)
        Returns:
            gradients dict with this step's parameter grads and da_prev,
            the hidden-state gradient to pass back one more step in time.
        """
        (a_prev, a_next, xt) = cache

        Waa = self.parameters["Waa"]
        # Backprop through the tanh non-linearity: d/dx tanh(x) = 1 - tanh(x)^2,
        # and a_next == tanh(...), so the derivative uses the *current* state a_next.
        dtanh = (1 - a_next ** 2) * da_next
        dWax = np.dot(dtanh, xt.T)                     # grad w.r.t. input weights
        dWaa = np.dot(dtanh, a_prev.T)                 # grad w.r.t. recurrent weights
        dba = np.sum(dtanh, axis=1, keepdims=True)     # grad w.r.t. hidden bias

        # Gradient flowing back to the previous time step's hidden state.
        da_prev = np.dot(Waa.T, dtanh)

        gradients = {"dWax": dWax,
                     "dWaa": dWaa,
                     "dba": dba,
                     "da_prev": da_prev}

        return gradients

    def rnn_backward(self, da, caches):
        """Backpropagation through time: sum the hidden-path gradients across steps.

        Args:
            da     : gradient on each hidden state coming from the output layer,
                     shape (n_a, m, T_x). The recurrent gradient flowing back from
                     future steps is added on top of this inside the loop.
            caches : (list_of_cell_caches, X) produced by rnn_forward.
        Returns:
            gradients dict accumulating dWax, dWaa, dba over the sequence.
            (dWya / dby are computed in train(), since the output layer is not
            part of the recurrent BPTT chain.)
        """
        (caches, X) = caches
        (a0, a1, x1) = caches[0]

        n_a, m, T_x = da.shape
        n_x, m = x1.shape                       # x1 is (n_x, m)

        # Accumulators for the parameter gradients (summed over time).
        dWax = np.zeros((n_a, n_x))             # input  -> hidden weights
        dWaa = np.zeros((n_a, n_a))             # hidden -> hidden (recurrent) weights
        dba = np.zeros((n_a, 1))                # hidden bias

        da_next = np.zeros((n_a, m))            # recurrent grad carried backward in time

        # Walk the sequence in reverse so gradients flow from the last step to the first.
        # At each step the total hidden gradient is the output-path grad da[:,:,t]
        # plus the recurrent grad da_next coming from the future.
        for t in reversed(range(T_x)):
            gradients = self.rnn_cell_backward(da[:, :, t] + da_next, caches[t])
            dWax += gradients["dWax"]
            dWaa += gradients["dWaa"]
            dba += gradients["dba"]
            da_next = gradients["da_prev"]      # pass hidden-state grad one step back

        gradients = {"dWax": dWax,
                     "dWaa": dWaa,
                     "dba": dba}

        return gradients

    def update_parameters(self, gradients):
        """Apply one gradient-descent step to every parameter."""
        Wax = self.parameters["Wax"]
        Waa = self.parameters["Waa"]
        Wya = self.parameters["Wya"]
        ba = self.parameters["ba"]
        by = self.parameters["by"]

        dWax = gradients["dWax"]
        dWaa = gradients["dWaa"]
        dba = gradients["dba"]
        dWay = gradients["dWya"]
        dby = gradients["dby"]

        # Move each parameter a small step in the direction that lowers the loss.
        Wax -= self.learning_rate * dWax
        Waa -= self.learning_rate * dWaa
        ba -= self.learning_rate * dba
        Wya -= self.learning_rate * dWay
        by -= self.learning_rate * dby

        parameters = {"Wax": Wax,
                      "Waa": Waa,
                      "Wya": Wya,
                      "ba": ba,
                      "by": by}

        self.parameters = parameters

    def train(self):
        """Full training loop: forward pass, loss, backprop, parameter update."""
        a0 = np.zeros((self.n_a, self.X.shape[1]))

        for i in range(self.iterations):
            caches, a, y_pred = self.rnn_forward(self.X, a0)      # forward pass

            # Output-layer gradient (same for softmax+CE and MSE): dy = y_pred - Y,
            # shape (n_y, m, T_x).
            dy = y_pred - self.Y

            # The output weights Wya / bias by are shared across every time step, so
            # their gradients sum over both the batch (m) and time (T_x) axes.
            # einsum 'imt,jmt->ij' = sum_t dy[:,:,t] @ a[:,:,t].T  -> (n_y, n_a).
            # np.dot equivalent (flatten the m and T_x axes into one, then 2D matmul):
            #   n_y, m, T_x = dy.shape
            #   dWya = np.dot(dy.reshape(n_y, m * T_x),
            #                 a.reshape(self.n_a, m * T_x).T)
            dWya = np.einsum("imt,jmt->ij", dy, a)                # grad w.r.t. output weights
            dby = np.sum(dy, axis=(1, 2)).reshape(self.n_y, 1)    # grad w.r.t. output bias

            # Gradient pushed from the output into each hidden state:
            # da[:,:,t] = Wya.T @ dy[:,:,t]  -> (n_a, m, T_x).
            # np.dot equivalent (flatten, 2D matmul, then reshape back):
            #   n_y, m, T_x = dy.shape
            #   da = np.dot(self.parameters["Wya"].T,
            #               dy.reshape(n_y, m * T_x)).reshape(self.n_a, m, T_x)
            da = np.einsum("ij,jmt->imt", self.parameters["Wya"].T, dy)

            # Backprop through time for the recurrent params (dWax, dWaa, dba).
            gradients = self.rnn_backward(da, caches)

            gradients.update({
                'dWya' : dWya,
                'dby' : dby
            })

            for key in gradients:
                np.clip(gradients[key], -5, 5, out=gradients[key])
                
            self.update_parameters(gradients)                     # gradient descent step

            # Log progress every 100 iterations.
            if i % 100 == 0:
                loss = self.compute_loss(y_pred, self.Y)              # how wrong are we?
                print("Iteration: {}, Loss: {}".format(i, loss))

    def predict(self, X):
        """Run a forward pass through the trained RNN and return predictions."""
        a0 = np.zeros((self.n_a, X.shape[1]))            # zero initial hidden state
        _, _, y_pred = self.rnn_forward(X, a0)           # forward returns (caches, a, y_pred)
        return y_pred

    def softmax(self, x):
        """Numerically stable softmax over axis 0 (subtract max before exp)."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
