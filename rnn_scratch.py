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
        # Weights and biases, created once and updated in place during training.
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

        return a_next, yt_pred

    def rnn_forward(self, X):
        """Run the forward pass over the whole sequence.

        Args:
            X : input sequence, shape (n_x, m, T_x)
        Returns:
            a      : every hidden state,  shape (n_a, m, T_x)
            y_pred : every prediction,    shape (n_y, m, T_x)
        """
        Wya = self.parameters["Wya"]

        n_x, m, T_x = X.shape
        n_y, n_a = Wya.shape

        a_next = np.zeros((n_a, m))            # initial hidden state a<0> = 0
        a = np.zeros((n_a, m, T_x))            # store hidden states for backprop
        y_pred = np.zeros((n_y, m, T_x))       # store predictions

        # Step through time, feeding each step's hidden state into the next.
        for t in range(T_x):
            xt = X[:, :, t]
            a_next, yt_pred = self.rnn_cell_forward(xt, a_next)
            a[:, :, t] = a_next
            y_pred[:, :, t] = yt_pred

        return a, y_pred

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

    def rnn_cell_backward(self, dy, da_next, xt, a_t, a_prev):
        """Backprop through one time step.

        Args:
            dy      : output-space gradient at this step (y_pred - Y), shape (n_y, m)
            da_next : hidden-state gradient flowing back from the *next* time step,
                      shape (n_a, m)
            xt      : input at this step,        shape (n_x, m)
            a_t     : hidden state at this step,  shape (n_a, m)
            a_prev  : hidden state from the previous step, shape (n_a, m)
        Returns:
            gradients dict with this step's parameter grads and da_prev,
            the hidden-state gradient to pass back one more step in time.
        """
        Waa = self.parameters["Waa"]
        Wya = self.parameters["Wya"]

        # --- output layer: z<t> = Wya @ a<t> + by, then softmax -> y<t> ---
        dWya = np.dot(dy, a_t.T)                       # grad w.r.t. output weights
        dby = np.sum(dy, axis=1, keepdims=True)        # grad w.r.t. output bias

        # Total gradient on this step's hidden state = gradient coming from the
        # output at this step, plus the gradient flowing back from future steps.
        da = np.dot(Wya.T, dy) + da_next               # shape (n_a, m)

        # Backprop through the tanh non-linearity: d/dx tanh(x) = 1 - tanh(x)^2,
        # and a_t == tanh(...), so the derivative uses the *current* state a_t.
        dtanh = (1 - a_t ** 2) * da
        dWax = np.dot(dtanh, xt.T)                     # grad w.r.t. input weights
        dWaa = np.dot(dtanh, a_prev.T)                 # grad w.r.t. recurrent weights
        dba = np.sum(dtanh, axis=1, keepdims=True)     # grad w.r.t. hidden bias

        # Gradient flowing back to the previous time step's hidden state.
        da_prev = np.dot(Waa.T, dtanh)

        gradients = {"dWax": dWax,
                     "dWaa": dWaa,
                     "dba": dba,
                     "dWya": dWya,
                     "dby": dby,
                     "da_prev": da_prev}

        return gradients

    def rnn_backward(self, y_pred, Y, a, X):
        """Backpropagation through time: sum gradients across all time steps.

        Args:
            y_pred : predictions at every step,  shape (n_y, m, T_x)
            Y      : one-hot targets at every step, shape (n_y, m, T_x)
            a      : hidden states at every step, shape (n_a, m, T_x)
            X      : the input sequence,          shape (n_x, m, T_x)
        Returns:
            gradients dict accumulating dWax, dWaa, dWya, dba, dby over the sequence.
        """
        Wax = self.parameters["Wax"]
        Waa = self.parameters["Waa"]
        Wya = self.parameters["Wya"]
        ba = self.parameters["ba"]
        by = self.parameters["by"]

        n_x, m, T_x = X.shape
        n_a = a.shape[0]

        # Accumulators for the parameter gradients (summed over time).
        dWax = np.zeros_like(Wax)
        dWaa = np.zeros_like(Waa)
        dWya = np.zeros_like(Wya)
        dba = np.zeros_like(ba)
        dby = np.zeros_like(by)

        da_next = np.zeros((n_a, m))  # hidden-state gradient carried backward in time

        # Walk the sequence in reverse so gradients flow from the last step to the first.
        for t in reversed(range(T_x)):
            # Gradient of softmax + cross-entropy at this step (output space).
            dy = y_pred[:, :, t] - Y[:, :, t]
            a_t = a[:, :, t]
            # Hidden state feeding into step t; a<-1> is the zero initial state.
            a_prev = a[:, :, t - 1] if t > 0 else np.zeros((n_a, m))

            gradients = self.rnn_cell_backward(dy, da_next, X[:, :, t], a_t, a_prev)
            dWax += gradients["dWax"]
            dWaa += gradients["dWaa"]
            dWya += gradients["dWya"]
            dba += gradients["dba"]
            dby += gradients["dby"]
            da_next = gradients["da_prev"]  # pass hidden-state gradient one step back

        gradients = {"dWax": dWax,
                     "dWaa": dWaa,
                     "dWya": dWya,
                     "dba": dba,
                     "dby": dby}

        # Clip gradients to a fixed range to prevent the "exploding gradient"
        # problem that vanilla RNNs suffer from over long backprop chains.
        for key in gradients:
            np.clip(gradients[key], -5, 5, out=gradients[key])

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
        for i in range(self.iterations):
            a, y_pred = self.rnn_forward(self.X)                  # forward pass
            loss = self.compute_loss(y_pred, self.Y)              # how wrong are we?
            gradients = self.rnn_backward(y_pred, self.Y, a, self.X)  # backprop through time
            self.update_parameters(gradients)                     # gradient descent step

            # Log progress every 100 iterations.
            if i % 100 == 0:
                print("Iteration: {}, Loss: {}".format(i, loss))

    def predict(self, X):
        """Run a forward pass through the trained RNN and return predictions."""
        _, y_pred = self.rnn_forward(X)
        return y_pred

    def softmax(self, x):
        """Numerically stable softmax over axis 0 (subtract max before exp)."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
