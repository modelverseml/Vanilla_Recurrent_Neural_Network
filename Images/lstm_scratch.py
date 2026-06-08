import numpy as np


class LSTM:

    def __init__(self, X, Y, n_a = 100, n_c=100, learning_Rate = 0.01, iterations = 1000,
                 task = 'classification'):


        self.X = X
        self.Y = Y
        self.n_a = n_a
        self.learning_rate = learning_Rate
        self.iteration = self.iteration

        if task not in ['classification', 'regression']:
            raise ValueError('task must be in classification or regression')
        
        self.task = task
        self.n_x = X.shape[0]
        self.n_y = Y.shape[0]
        
        self.parameters = self.initialize_parameters()
    
    def initialize_parameters(self):

        np.random.seed(1)

        
        Wf = np.random.randn(self.n_a, self.n_a+self.n_x)
        bf = np.zeros((self.n_a, 1))

        Wi = np.random.randn(self.n_a, self.n_a+self.n_x)
        bi = np.zeros((self.n_a, 1))

        Wc = np.random.randn(self.n_a, self.n_a+self.n_x)
        bc = np.zeros((self.n_a, 1))

        Wo = np.random.randn(self.n_a, self.n_a+self.n_x)
        bo = np.zeros((self.n_a, 1))

        Wy = np.random.randn(self.n_a, self.n_a+self.n_x)
        by = np.zeros((self.n_a, 1))


        parameters = {
            'Wf' : Wf,
            'bf' : bf,
            'Wi' : Wi,
            'bi' : bi,
            'Wc' : Wc,
            'bc' : bc,
            'Wo' : Wo,
            'bo' : bo,
            'Wy' : Wy,
            'by' : by
        }

        return parameters
    

    
    def run_cell_forward(self, xt, a_prev, c_prev):

        Wf = self.parameters['Wf']
        bf = self.parameters['bf']
        Wi = self.parameters['Wi']
        bi = self.parameters['bi']
        Wc = self.parameters['Wc']
        bc = self.parameters['bc']
        Wo = self.parameters['Wo']
        bo = self.parameters['bo']
        Wy = self.parameters['Wy']
        by = self.parameters['by']


        x_concat = np.concatenate([a_prev, xt],axis=0)
        gamma_f = self.sigmoid(np.dot(Wf, x_concat) + bf)
        gamma_i = self.sigmoid(np.dot(Wi, x_concat) + bi)

        cct = np.tanh(np.dot(Wc, x_concat) + bc)
        c_next = gamma_f * c_prev + gamma_i * cct

        gamma_o = self.sigmoid(np.dot(Wo, x_concat) + bo)
        a_next = gamma_o * np.tanh(c_next)
        yt_pred = self.softmax(np.dot(Wy, a_next) + by)


        cache = (a_next, c_next, a_prev, c_prev, gamma_f, gamma_i, gamma_o, cct, xt)

        return cache, a_next, c_next, yt_pred


    def run_forward(self, X):

        Wya = self.parameters["Wya"]

        n_x, m, T_x = X.shape
        n_y, n_a = Wya.shape

        cache = []

        a_next = np.zeros((n_a, m))            # initial hidden state a<0> = 0
        c_next = np.zeros((n_a, m))
        a = np.zeros((n_a, m, T_x))            # store hidden states for backprop
        c = np.zeros((n_a, m, T_x))
        y_pred = np.zeros((n_y, m, T_x))       # store predictions

        # Step through time, feeding each step's hidden state into the next.
        for t in range(T_x):
            xt = X[:, :, t]
            cache, a_next, c_next, yt_pred = self.run_cell_forward(xt, a_next, c_next)
            cache.append(cache)
            a[:, :, t] = a_next
            c[:, :, t] = c_next
            y_pred[:, :, t] = yt_pred

        return a,c, y_pred
    
    def compute_loss(self, y_pred, Y):

        m = Y.shape[1]
        if self.task == "classification":
            # Clip to avoid log(0) -> -inf / nan when a probability underflows.
            y_pred = np.clip(y_pred, 1e-12, 1.0)
            return -np.sum(Y * np.log(y_pred)) / m
        # regression: mean squared error (the 1/2 makes the gradient exactly y_pred - Y)
        return 0.5 * np.sum((y_pred - Y) ** 2) / m
    

    def cell_backward(self, dy,xt,):

        Wf = self.parameters['Wf']
        bf = self.parameters['bf']
        Wi = self.parameters['Wi']
        bi = self.parameters['bi']
        Wc = self.parameters['Wc']
        bc = self.parameters['bc']
        Wo = self.parameters['Wo']
        bo = self.parameters['bo']
        Wy = self.parameters['Wy']
        by = self.parameters['by']

        dot = da_next * np.tanh(c_next) * self.sigmoid(np.dot(Wo, x_concat) + bo) * (1 - self.sigmoid(np.dot(Wo, x_concat) + bo))



    def sigmoid(self, x):

        return 1/(1 - np.exp(-x))
    
    def softmax(self, x):
        """Numerically stable softmax over axis 0 (subtract max before exp)."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)



