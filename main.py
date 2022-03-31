import time
from functools import partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from model.SVM import SoftMarginKernelSVM
from model.algorithm.kernels import no_kernel, polynomial_kernel, gaussian_kernel
from model.algorithm.utils import draw
from sklearn.svm import SVC, LinearSVC

datasets = [
    # "linear_hard_margin.csv",
    "Circle.csv"
]

# Init
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(10, 15))
# fig.tight_layout()

data = pd.read_csv(datasets[0])
X, Y = data.iloc[:, :-1], data.iloc[:, -1]

# 软间隔 L1 无核函数 SVM
begin = time.time()
model = SoftMarginKernelSVM(kernel=no_kernel)
model.fit(X.values, Y.values[:, np.newaxis])
end = time.time()
draw(model, X, Y, ax1, f"My L1 No-kernel,t={round(end - begin, 2)}")

# Sklearn
begin = time.time()
model = LinearSVC(penalty='l1', loss="squared_hinge", dual=False)
model.fit(X, Y)
decision_function = model.decision_function(X)
support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
support_vectors = X.iloc[support_vector_indices]
end = time.time()
draw(model, X, Y, ax2, f"Sklearn L1 No-Kernel,t={round(end - begin, 2)}", support_vectors=support_vectors.values)

# 软间隔 L2 无核函数 SVM
begin = time.time()
model = SoftMarginKernelSVM(kernel=no_kernel, C=1e10)
model.fit(X.values, Y.values[:, np.newaxis])
end = time.time()
draw(model, X, Y, ax3, f"My L2 No-kernel,t={round(end - begin, 2)}")

# SKlearn L2 无核函数
begin = time.time()
model = LinearSVC(penalty='l2')
model.fit(X, Y)
decision_function = model.decision_function(X)
support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
support_vectors = X.iloc[support_vector_indices]
end = time.time()
draw(model, X, Y, ax4, f"Sklearn L2 No-Kernel,t={round(end - begin, 2)}", support_vectors=support_vectors.values)

# 软间隔 多项式核函数 SVM
begin = time.time()
model = SoftMarginKernelSVM(kernel=partial(polynomial_kernel, c=2, degree=5))
model.fit(X.values, Y.values[:, np.newaxis])
end = time.time()
draw(model, X, Y, ax5, f"My Polynomial-kernel,t={round(end - begin, 2)}")

# Sklearn 多项式核函数
begin = time.time()
model = SVC(kernel='poly')
model.fit(X, Y)
end = time.time()
draw(model, X, Y, ax6, f"Sklearn Polynomial-kernel,t={round(end - begin, 2)}")

# 软间隔 高斯核函数 SVM
begin = time.time()
model = SoftMarginKernelSVM(kernel=partial(gaussian_kernel, c=2))
model.fit(X.values, Y.values[:, np.newaxis])
end = time.time()
draw(model, X, Y, ax7, f"My Gauss-kernel,t={round(end - begin, 2)}")

# Sklearn 高斯核函数
begin = time.time()
model = SVC(kernel='rbf')
model.fit(X, Y)
end = time.time()
draw(model, X, Y, ax8, f"Sklearn Gauss-kernel,t={round(end - begin, 2)}")

plt.show()
fig.savefig(f"Last-{datasets[0]}.png")
fig.savefig(f"Last.png")
