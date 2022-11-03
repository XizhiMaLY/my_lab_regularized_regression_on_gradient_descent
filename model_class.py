import numpy as np


def minimize(B, x, y, eta, max_iter, precision):
    while np.linalg.norm(grad := grad_linear(x, y, B)) > precision and max_iter>0:
        B = B - eta*grad
        max_iter -= 1
    return B


def grad_linear(x, y, B):
    return -np.dot(np.transpose(x), y - np.dot(x, B))


class Linear_mylab:
    def __init__(self, eta=0.00001, max_iter=1000, precision=1e-9):
        self.eta = eta
        self.max_iter = max_iter
        self.precision = precision


    def fit(self,x,y):
        p = len(x[0])
        B = np.random.uniform(-1,1,p)
        B = minimize(B, x, y, self.eta, self.max_iter, self.precision)
        return B