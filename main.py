import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from model.SVM import SoftMarginKernelSVM

from model.algorithm.kernels import no_kernel

data = pd.read_csv("linear_hard_margin.csv")
model = SoftMarginKernelSVM(kernel=no_kernel)
X, Y = data.iloc[:, :-1], data.iloc[:, -1]
result = model.fit(X.values, Y.values[:, np.newaxis])
ax = plt.gca()
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=Y, s=50, cmap='autumn')
# 获取上下限
x_lim = ax.get_xlim()
y_lim = ax.get_ylim()
# 准备作直线
xx = np.linspace(x_lim[0], x_lim[1], 30)
yy = np.linspace(y_lim[0], y_lim[1], 30)
# https://www.cnblogs.com/lemonbit/p/7593898.html
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()
