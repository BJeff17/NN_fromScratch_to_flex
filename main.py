import numpy as np
import json

class NN:
    def __init__(self, nn_config=[2, 8, 16, 1]):
        self.params = {}
        self.grads = {}
        self.nn_config = nn_config
        self.n_layer = len(nn_config) - 1
        self.cache = {}

    def init_params(self):
        for idx in range(self.n_layer):
            self.params[f"w{idx+1}"] = np.random.randn(self.nn_config[idx], self.nn_config[idx+1]) * 0.01
            self.params[f"b{idx+1}"] = np.zeros((1, self.nn_config[idx+1]))
            self.grads[f"w{idx+1}"] = np.zeros((self.nn_config[idx], self.nn_config[idx+1]))
            self.grads[f"b{idx+1}"] = np.zeros((1, self.nn_config[idx+1]))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return z > 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        a = X
        self.cache['a0'] = X
        for idx in range(self.n_layer):
            w = self.params[f"w{idx+1}"]
            b = self.params[f"b{idx+1}"]
            z = np.dot(a, w) + b
            if idx < self.n_layer - 1:
                a = self.relu(z)
            else:
                a = self.sigmoid(z)
            self.cache[f"z{idx+1}"] = z
            self.cache[f"a{idx+1}"] = a
        return a

    def calc_loss_nll(self, y_pred, y_true):
        loss = -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
        return loss

    def calc_grad_nll(self, y_true):
        m = y_true.shape[0]
        dz = self.cache[f"a{self.n_layer}"] - y_true  
        for idx in reversed(range(1, self.n_layer + 1)):
            a_prev = self.cache[f"a{idx-1}"]
            self.grads[f"w{idx}"] = (1/m) * np.dot(a_prev.T, dz)
            self.grads[f"b{idx}"] = (1/m) * np.sum(dz, axis=0, keepdims=True)

            if idx > 1:
                da_prev = np.dot(dz, self.params[f"w{idx}"].T)
                dz = da_prev * self.relu_deriv(self.cache[f"z{idx-1}"])

    def update_weight(self, lr):
        for idx in range(1, self.n_layer + 1):
            self.params[f"w{idx}"] -= lr * self.grads[f"w{idx}"]
            self.params[f"b{idx}"] -= lr * self.grads[f"b{idx}"]

    def save_params_json(self, save="params.json"):
        json_ready = {k: v.tolist() for k, v in self.params.items()}
        with open(save, "w") as f:
            json.dump(json_ready, f)

    def load_params_json(self, save="params.json"):
        try:
            with open(save, "r") as f:
                loaded = json.load(f)
                self.params = {k: np.array(v) for k, v in loaded.items()}
        except:
            print("Problème durant l'ouverture du fichier")
