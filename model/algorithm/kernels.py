import numpy as np
import numexpr as ne
from scipy.linalg.blas import dgemm, sgemm


def no_kernel(x, y: np.ndarray):
    return x @ y.T


def polynomial_kernel(P, Q, c, degree):
    return ne.evaluate('(c + A)**d', {
        'A': dgemm(alpha=1.0, a=P, b=Q, trans_b=True),
        'c': c,
        'd': degree
    })


def gaussian_kernel(P, Q, c):
    P_norm = np.einsum('ij,ij->i', P, P)
    Q_norm = np.einsum('ij,ij->i', Q, Q)
    return ne.evaluate('exp(-gamma * (A + B - 2*C))', {
        'A': P_norm[:, None],
        'B': Q_norm[None, :],
        'C': dgemm(alpha=1.0, a=P, b=Q, trans_b=True),
        'gamma': c
    })
