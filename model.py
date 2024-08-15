import numpy as np

class deepNeuralNetwork:
    def __init__(self, size, activation):
        self.size = size
        self.params = self.initialize()
        self.cache = {}
        if activation == "relu":
            self.activation = self.relu
        if activation == "sigmoid":
            self.activation = self.sigmoid
        self.momentum = {key: np.zeros_like(val) for key, val in self.params.items()}
        self.momsq = {key: np.zeros_like(val) for key, val in self.params.items()}
        self.t = 0

    def initialize(self):
        input_layer = self.size[0]
        hidden_layer = self.size[1]
        output_layer = self.size[2]
        params = {
            "w1": np.random.randn(hidden_layer, input_layer) * np.sqrt(1. / input_layer),
            "b1": np.random.randn(hidden_layer, 1) * np.sqrt(1. / input_layer),
            "w2": np.random.randn(output_layer, hidden_layer) * np.sqrt(1. / hidden_layer),
            "b2": np.random.randn(output_layer, 1) * np.sqrt(1. / hidden_layer)
        }
        return params

    def sigmoid(self, x, derivative=False):
        if derivative:
            sig = self.sigmoid(x)
            return sig * (1 - sig)
        return 1 / (1 + np.exp(-x))

    def relu(self, x, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)

    def softmax(self, x):
        ex = np.exp(x - x.max(axis=0, keepdims=True))
        return ex / np.sum(ex, axis=0, keepdims=True)

    def cross_entropy_loss(self, y, y_pred):
        m = y.shape[0]
        loss = -np.sum(y * np.log(y_pred.T + 1e-8)) / m  # Transpose y_pred to match y's shape and add epsilon to avoid log(0)
        return loss

    def forward_prop(self, x):
        self.cache["x"] = x
        self.cache["z1"] = np.matmul(self.params["w1"], self.cache["x"].T) + self.params["b1"]
        self.cache["a1"] = self.activation(self.cache["z1"])
        self.cache["z2"] = np.matmul(self.params["w2"], self.cache["a1"]) + self.params["b2"]
        self.cache["a2"] = self.softmax(self.cache["z2"])
        return self.cache["a2"]

    def back_prop(self, y):
        m = y.shape[0]
        dz2 = self.cache["a2"] - y.T
        dw2 = (1 / m) * np.matmul(dz2, self.cache["a1"].T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        da1 = np.matmul(self.params["w2"].T, dz2)
        dz1 = da1 * self.activation(self.cache["z1"], derivative=True)
        dw1 = (1 / m) * np.matmul(dz1, self.cache["x"])
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
        self.grads = {"w2": dw2, "b2": db2, "w1": dw1, "b1": db1}
        return self.grads

    def optimize(self, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, optimizer="gradient_descent"):
        self.optimizer = optimizer
        if optimizer == "gradient_descent":
            for key in self.params:
                self.params[key] -= alpha * self.grads[key]
        elif optimizer == "momentum":
            for key in self.params:
                self.momentum[key] = beta1 * self.momentum[key] + (1 - beta1) * self.grads[key]
                self.params[key] -= alpha * self.momentum[key]
        elif optimizer == "RMSprop":
            for key in self.params:
                self.momsq[key] = beta2 * self.momsq[key] + (1 - beta2) * (self.grads[key] ** 2)
                self.params[key] -= alpha * self.grads[key] / (np.sqrt(self.momsq[key]) + epsilon)
        elif optimizer == "adam":
            self.t += 1
            for key in self.params:
                self.momentum[key] = beta1 * self.momentum[key] + (1 - beta1) * self.grads[key]
                self.momsq[key] = beta2 * self.momsq[key] + (1 - beta2) * (self.grads[key] ** 2)
                m_hat = self.momentum[key] / (1 - beta1 ** self.t)
                v_hat = self.momsq[key] / (1 - beta2 ** self.t)
                self.params[key] -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

    def accuracy(self, y, y_pred):
        y_pred_labels = np.argmax(y_pred, axis=0)
        y_true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(y_pred_labels == y_true_labels)
        return accuracy

    def train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=64, optimizer="gradient_descent"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        for i in range(self.epochs):
            epoch_loss = 0
            num_batches = x_train.shape[0] // self.batch_size
            for j in range(num_batches):
                begin = j * self.batch_size
                end = begin + self.batch_size
                x = x_train[begin:end]
                y = y_train[begin:end]
                output = self.forward_prop(x)
                grads = self.back_prop(y)
                self.optimize(alpha=0.01, optimizer=self.optimizer)
                batch_loss = self.cross_entropy_loss(y, output)
                epoch_loss += batch_loss
            epoch_loss /= num_batches
            test_output = self.forward_prop(x_test)
            test_loss = self.cross_entropy_loss(y_test, test_output)
            train_output = self.forward_prop(x_train)
            train_accuracy = self.accuracy(y_train, train_output)
            test_accuracy = self.accuracy(y_test, test_output)
            print(f'Epoch {i + 1}/{self.epochs}, Training Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
        print('Training complete.')
