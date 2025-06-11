
import matplotlib.pyplot as plt
import numpy as np

from main import NN

def generate_data(n=100):
    X = np.random.randn(n, 2)
    y = (X[:, 0] * X[:, 1] > 0).astype(float).reshape(-1, 1)
    return X, y

if __name__ == "__main__":
    nn = NN([2,4,1])
    nn.init_params()
    
    X, y = generate_data(200)
    losses = []

    for epoch in range(1000000):#I love see overfitting...
        y_pred = nn.forward(X)
        loss = nn.calc_loss_nll(y_pred, y)
        losses.append(loss)
        nn.calc_grad_nll(y)
        nn.update_weight(0.01)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="Loss", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Évolution de la perte pendant l'entraînement")
    plt.legend()
    plt.grid(True)
    plt.show()

    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    Z = nn.forward(grid_points)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, Z, cmap="bwr", alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolor='k', cmap="bwr", s=40)
    plt.title("Frontière de décision")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.show()
