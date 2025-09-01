import numpy as np
from numpy import linalg

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

def handle_target(c, target):
    if target == 'max':
        return -c
    elif target == 'min':
        return c
    else:
        raise ValueError("Target must be 'max' or 'min'")

def parse(problem, z, target='max'):
    num_constraints, num_vars = problem.shape[0], problem.shape[1] - 2

    A = np.zeros((num_constraints, num_vars + num_constraints))
    b = np.zeros((num_constraints, 1))
    c = np.zeros((num_vars + num_constraints, 1))

    for i in range(num_constraints):
        A[i, :num_vars] = problem[i, :num_vars]
        b[i] = problem[i, -1]
        if problem[i, -2] == '<=':
            A[i, num_vars + i] = 1
        elif problem[i, -2] == '>=':
            A[i, num_vars + i] = -1
        else:
            raise ValueError("Inequality must be '<=' or '>='")

    c[:num_vars] = z[:, None]
    c = handle_target(c, target)


    Xb = np.array(range(num_vars, num_vars + num_constraints))
    Xn = np.array(range(num_vars))

    return A, b, c, Xn, Xb

target = 'max'
problem = np.array([
    [1, 1, '<=', 4],
    [1, 0, '<=', 3],
    [0, 1, '<=', 2]
])
z = np.array([5, 2])


A, b, c, Xn, Xb = parse(problem, z, target)

simplex_min(A, b, c, Xn, Xb)
