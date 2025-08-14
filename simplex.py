import numpy as np
from numpy import linalg

c = np.array([-5, -3, 0, 0])
A = np.array([[3, 5, 1, 0], [5, 2, 0, 1]])
b = np.array([15, 10])[:, None]
Ni = np.array([1, 2]) - 1
Bi = np.array([3, 4]) - 1


def simplex_min(c, A, b, Ni, Bi):
    B = [x[Bi] for x in A]
    iters = 0
    while True:
        iters += 1
        if iters > 10:
            return
        cb = c[Bi]
        B_inv = linalg.inv(B)
        xBarra = np.diag(B_inv * b)[:, None]
        zBarra = cb * xBarra * b
        max_dif = -np.inf
        k = None
        for j in Ni:
            cur_dif = np.diag((cb * B_inv - c[j]))[0]
            if cur_dif > max_dif:
                max_dif = cur_dif
                k = j
        if max_dif < 0:
            return zBarra
        ai = [x[k] for x in A]
        y = np.diag(B_inv * ai)[:, None]
        xb = np.inf
        xbi = None
        for yi in range(np.size(y)):
            if y[yi] == 0:
                pass
            cur_xb = xBarra[yi] / y[yi]
            if cur_xb < xb:
                xb = cur_xb
                xbi = yi
        Bi[xbi] = xbi
        B[xbi] = ai


simplex_min(c, A, b, Ni, Bi)
