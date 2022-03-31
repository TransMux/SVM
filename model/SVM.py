from typing import Callable

import numpy as np

from model.algorithm.kernels import no_kernel
from model.algorithm.smo import SMO


class SoftMarginKernelSVM:
    def __init__(self, kernel: Callable[[np.ndarray, np.ndarray], np.ndarray] = no_kernel, solver=SMO, C=1.0):
        self.kernel = kernel
        self.solver = solver(C)
        self.X = None
        self.Y = None
        self.alpha = None
        self.b = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y
        kernel_matrix = self.kernel(X, X)
        self.alpha, self.b = self.solver.solve(kernel_matrix, Y)
        self.alpha = self.alpha.squeeze()

    def predict(self, X, raw=False):
        # m * f, n * f -> m * n
        kernel_matrix = self.kernel(self.support_vectors_, X)
        # m * 1,
        raw_predicts = np.squeeze((self.support_vectors_y.T * self.support_alphas) @ kernel_matrix + self.b)
        if raw:
            return raw_predicts
        raw_predicts[raw_predicts > 0] = 1
        raw_predicts[raw_predicts < 0] = -1
        return raw_predicts

    @property
    def support_vectors_(self):
        return self.X[self.alpha > 0]

    @property
    def support_vectors_y(self):
        return self.Y[self.alpha > 0]

    @property
    def support_alphas(self):
        return self.alpha[self.alpha > 0]

    def decision_function(self, x: np.ndarray):
        return self.predict(x, raw=True)
