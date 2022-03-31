from functools import partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from model.SVM import SoftMarginKernelSVM
from model.algorithm.kernels import no_kernel, polynomial_kernel
from model.algorithm.utils import draw
from sklearn.svm import SVC, LinearSVC

# Init

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 10))
data = pd.read_csv("linear_hard_margin.csv")
X, Y = data.iloc[:, :-1], data.iloc[:, -1]

# 软间隔 L1 无核函数 SVM
model = SoftMarginKernelSVM(kernel=no_kernel)
model.fit(X.values, Y.values[:, np.newaxis])
draw(model, X, Y, ax1, "My L1 No-kernel")

# Sklearn
model = LinearSVC(penalty='l1', loss="squared_hinge", dual=False)
model.fit(X, Y)
decision_function = model.decision_function(X)
support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
support_vectors = X.iloc[support_vector_indices]
draw(model, X, Y, ax2, "Sklearn L1 No-Kernel", support_vectors=support_vectors.values)

# 软间隔 L2 无核函数 SVM
model = SoftMarginKernelSVM(kernel=no_kernel)
model.fit(X.values, Y.values[:, np.newaxis])
draw(model, X, Y, ax3, "My L2 No-kernel")

# SKlearn L2 无核函数
model = LinearSVC(penalty='l2')
model.fit(X, Y)
decision_function = model.decision_function(X)
support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
support_vectors = X.iloc[support_vector_indices]
draw(model, X, Y, ax4, "Sklearn L2 No-Kernel", support_vectors=support_vectors.values)

# 软间隔 多项式核函数 SVM
model = SoftMarginKernelSVM(kernel=partial(polynomial_kernel, c=2, degree=5))
model.fit(X.values, Y.values[:, np.newaxis])
draw(model, X, Y, ax5, "My Polynomial-kernel")

plt.show()
fig.savefig("Last.png")
