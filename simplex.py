import numpy as np
from numpy import linalg


def log(msg, verbose=True):
    if verbose:
        print(msg)


def simplex_min(A, b, c, Xn, Xb, verbose=True):
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
            log("\n***Optimal solution found!***", verbose)
            log(f"Optimal value: {zBarra.flatten()[0]}", verbose)
            log(f"Basic variables: {Xb}, Non-basic variables: {Xn}", verbose)
            log(f"Solution: {xBarra.flatten()}", verbose)
            return Xb, xBarra, zBarra

        yi = B_inv @ A[:, Xn[imax]]

        with np.errstate(divide="ignore", invalid="ignore"):
            theta = np.where(yi > 0, xBarra.flatten() / yi.flatten(), np.inf)

        if np.all(theta == np.inf):
            log("Unbounded solution", verbose)
            return None

        imin = np.argmin(theta)

        Xb[imin], Xn[imax] = Xn[imax], Xb[imin]
        log(
            f"Iteration {_ + 1}: Xb = {Xb}, xBarra = {xBarra.flatten()}, zBarra = {zBarra.flatten()}",
            verbose,
        )

    log("Maximum iterations reached without finding optimal solution", verbose)

    return None


def handle_target(c, target):
    if target == "max":
        return -c
    elif target == "min":
        return c
    else:
        raise ValueError("Target must be 'max' or 'min'")


def parse(z, constraints, target="max"):
    num_constraints, num_vars = constraints.shape[0], constraints.shape[1] - 2

    A = np.zeros((num_constraints, num_vars + num_constraints))
    b = np.zeros((num_constraints, 1))
    c = np.zeros((num_vars + num_constraints, 1))

    for i in range(num_constraints):
        A[i, :num_vars] = constraints[i, :num_vars]
        b[i] = constraints[i, -1]
        if constraints[i, -2] == "<=":
            A[i, num_vars + i] = 1
        elif constraints[i, -2] == ">=":
            A[i, num_vars + i] = -1
        else:
            raise ValueError("Inequality must be '<=' or '>='")

    c[:num_vars] = z[:, None]
    c = handle_target(c, target)

    Xb = np.array(range(num_vars, num_vars + num_constraints))
    Xn = np.array(range(num_vars))

    return A, b, c, Xn, Xb


def simplex(z, constraints, target, verbose=True):
    A, b, c, Xn, Xb = parse(z, constraints, target)

    two_phase_needed = np.any(constraints[:, -2] == ">=") or np.any(b < 0)
    if two_phase_needed:
        log("Two-phase method needed. Not implemented.", verbose)
        return None

    return simplex_min(A, b, c, Xn, Xb, verbose)


if __name__ == "__main__":
    target = "max"
    constraints = np.array([[1, 1, "<=", 4],
                            [1, 0, "<=", 3],
                            [0, 1, "<=", 2]])
    z = np.array([5, 2])

    simplex(z, constraints, target, True)
