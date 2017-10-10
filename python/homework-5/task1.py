#!/usr/bin/env python3
# encoding: utf-8

# System imports.
import sys
import time

# Numerical analysis imports.
import numpy as np
import numpy.random as nr
import scipy.linalg as sl

# Other package imports.
import tabulate

def qrmwrsad(A, apply_hessenberg=True, rtol=1e-8):
    """
    Compute the eigenvalues of the square matrix A by applying the QR
    method with Raileigh shifts and deflations.

    The matrix can be put in Hessenberg form before applying the
    iteration and recursion, in order to improve convergence. This is
    the default, but is configurable since the algorithm recurses and
    upon recursion the matrix is already Hessenberg. The algorithm makes
    the assumption that the matrix is Hessenberg, so only change this if
    you are willing to take the consequences such as bad precision or
    poor convergence.

    :param A: the matrix whose eigenvalues to compute
    :type A: np.ndarray
    :param apply_hessenberg: whether or not to bring the matrix into
                             Hessenberg form before iterating.
    :type apply_hessenberg: bool
    :param rtol: relative tolerance; the maximum relative error of the
                 returned eigenvalues
    :param rtol: float
    :returns: a tuple containing firstly a list of eigenvalues and
              secondly the number of iterations that was required to get
              the result
    """
    m = A.shape[0]

    # Handle the trivial case of a 1x1 matrix. This guarantees that the
    # deflation ends.
    if m == 1:
        return [A[0,0]], 0

    if apply_hessenberg:
        A = sl.hessenberg(A)

    # Keep track of the iteration count as required by the task.
    iteration_count = 0

    # Boolean array representing the superdiagonal elements.
    super_diagonal = np.tri(m, k=1, dtype=bool) - np.tri(m, dtype=bool)

    # Continue iterating as long as no superdiagonal element is close to
    # zero.
    #
    # Note that the tolerance needs to be split in two, because by the
    # Gerschgorin Circle Theorem if the off-diagonal elements are less
    # that the half the tolerance then the eigenvalue is off by atmost
    # the tolerance, under the assumption that the matrix is
    # tridiagonal, which is true if A is symmetric and has been brought
    # to Hessenberg form.
    isclose = np.isclose(A[super_diagonal], 0, rtol=rtol/2)
    while not isclose.any():
        iteration_count += 1

        mu = A[m-1,m-1]
        muI = mu * np.eye(m)

        # Since the matrix is Hessenberg this QR factorization can be
        # optimized by Givens rotations, but let's not do that.
        Q, R = sl.qr(A - muI)
        A = np.dot(R, Q) + muI
        isclose = np.isclose(A[super_diagonal], 0, rtol=rtol/2)

    if not isclose.all():
        # If not all superdiagonal entries are close to zero, find one
        # such and deflate at it.
        for j in range(m-1):
            if isclose[j]:
                break

        # There is no need set these entries to zero since they will not
        # be considered in the deflation anyway.
        # A[j,j+1] = 0
        # A[j+1,j] = 0

        # Perform the deflation on the submatrices (recurse). It is not
        # necessary for the recursive call to bring the matrix to
        # Hessenberg form since it already is.
        return_list = []
        for B in [A[:j+1,:j+1], A[j+1:,j+1:]]:
            l,i = qrmwrsad(B, apply_hessenberg=False)
            return_list += l
            iteration_count += i
        return return_list, iteration_count
    else:
        # If all superdiagonal elements are close to zero
        diagonal = np.diag([True]*m)
        return list(A[diagonal]), iteration_count

if __name__ == '__main__':
    def random_symmetric_matrix(m):
        """
        Generate a symmetric random mxm matrix.

        :param m: the desired number of rows and columns
        :type m: integer
        """
        A = nr.rand(m, m)
        return (A + A.T)/2

    table = [[
        "m",
        "Mean iteration count",
        "Our mean",
        "Our min",
        "Out max",
        "Scipy mean",
        "Scipy min",
        "Scipy max"
    ]]

    MIN_M = 64
    MAX_M = 512
    STEP_M = 64
    N_ITERATIONS = 64
    for m in range(MIN_M, MAX_M+1, STEP_M):

        # Save the computation times so that some statistics can be
        # printed in the table.
        our_times = []
        sls_times = []
        iteration_counts = []

        for i in range(N_ITERATIONS):
            # Print the status nicely. This does not flood the terminal,
            # but contains all necessary information.
            sys.stdout.write("\r")
            sys.stdout.write("{} / {}: {} / {}"
                    .format(m, MAX_M, i + 1, N_ITERATIONS)
                    .ljust(72))
            sys.stdout.flush()

            A = random_symmetric_matrix(m)

            clk1 = time.clock()
            our_eigenvalues, iteration_count = qrmwrsad(A)
            clk2 = time.clock()
            our_time = clk2 - clk1

            clk1 = time.clock()
            sls_eigenvalues = sl.eig(A)[0]
            clk2 = time.clock()
            sls_time = clk2 - clk1

            # Collect statistics.
            our_times.append(our_time)
            sls_times.append(sls_time)
            iteration_counts.append(iteration_count)

            # Sort the eigenvalues so that they can be compared. This is
            # necessary because there is no guarantee that the returned
            # list and array respectively are sorted the same.
            our_eigenvalues.sort()
            sls_eigenvalues.sort()
            if not np.isclose(our_eigenvalues, sls_eigenvalues).all():
                print()
                print("The eigenvalues differ!")
                print(A)

        row = [m]

        # Append the average iteration count to the row.
        row.append(sum(iteration_counts) / len(iteration_counts))

        # Append the statistics from the two times lists to the result
        # row.
        def append_times(times):
            row.append(sum(times) / len(times))
            row.append(min(times))
            row.append(max(times))
        append_times(our_times)
        append_times(sls_times)

        table.append(row)

    print()
    with open("task1.tex", "w") as f:
        f.write(tabulate.tabulate(table, headers="firstrow",
                tablefmt="latex"))
