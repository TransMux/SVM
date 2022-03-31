import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from model.SVM import SoftMarginKernelSVM
from model.algorithm.kernels import no_kernel
from model.algorithm.utils import draw
from sklearn.svm import SVC

# Init

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
data = pd.read_csv("linear_hard_margin.csv")
X, Y = data.iloc[:, :-1], data.iloc[:, -1]

# 软间隔 L1 无核函数 SVM
model = SoftMarginKernelSVM(kernel=no_kernel)
model.fit(X.values, Y.values[:, np.newaxis])
draw(model, X, Y, ax1, "My L1 No-kernel SVM")

# Sklearn
model = SVC(kernel='linear', C=1E10)
model.fit(X, Y)
draw(model, X, Y, ax2, "Sklearn L1 No-Kernel")

# 软间隔 L2 无核函数 SVM
model = SoftMarginKernelSVM(kernel=no_kernel)
model.fit(X.values, Y.values[:, np.newaxis])
draw(model, X, Y, ax3, "My L2 No-kernel SVM")



plt.show()
fig.savefig("Last.png")
