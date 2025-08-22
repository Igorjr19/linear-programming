import numpy as np
from numpy import linalg

# Matriz dos coeficientes das inequações
A = np.array([
    [1, 1, 1, 0, 0], 
    [1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1]
    ]) 

# "Respostas" das inequações
b = np.array([4, 3, 2])[:, None]  

# Função objetivo
c = np.array([-5, -2, 0, 0, 0])[:, None]  

# Variáveis básicas e não básicas
Xb = np.array([2, 3, 4])
Xn = np.array([0, 1])


def simplex_min(A, b, c, Xn, Xb):
    max_iter = 10

    for _ in range(max_iter):
        B = A[:, Xb]
        B_inv = linalg.inv(B)
        cb = c[Xb]
        cn = c[Xn]

        xBarra = B_inv @ b
        zBarra = cb.T @ xBarra

        zj_cj = cb.T @ B_inv @ A[:, Xn] - cn.T
        zj_cj = zj_cj.flatten()
        imax = np.argmax(zj_cj)
        if zj_cj[imax] <= 0:
            print("\n***Optimal solution found!***")
            print(f"Optimal value: {zBarra.flatten()[0]}")
            print(f"Basic variables: {Xb}, Non-basic variables: {Xn}")
            print(f"Solution: {xBarra.flatten()}")
            return Xb, xBarra, zBarra
        
        yi = B_inv @ A[:, Xn[imax]]
        theta = xBarra.T / yi
        if np.all(theta <= 0):
            print("Unbounded solution")
            return None
        
        imin = np.argmin(theta[theta > 0])

        Xb[imin], Xn[imax] = Xn[imax], Xb[imin]
        print(f"Iteration {_ + 1}: Xb = {Xb}, xBarra = {xBarra.flatten()}, zBarra = {zBarra.flatten()}")
        


    print("Maximum iterations reached without finding optimal solution")
    
    return None


simplex_min(A, b, c, Xn, Xb)
