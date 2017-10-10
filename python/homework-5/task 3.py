# encoding: utf-8
import sys
import numpy as np
import numpy.random as nr
import scipy.linalg as sl

def count_eig_negative(A, x):
    m = A.shape[0]
    p0 = 1
    p1 = A[0, 0] - x
    n = int(p1 <= 0)
    for k in range(1, m):
        p2 = (A[k, k]-x) * p1 - A[k - 1, k] ** 2 * p0
        n += int(p1 * p2 <= 0)
        p0 = p1
        p1 = p2
    return n

def count_eig_between(A, a, b):
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
    isclose = np.isclose(A[super_diagonal], 0)
    
    if isclose.any():
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
            l,i = qrmwrsad(B)
            return_list += l
            iteration_count += i
        return return_list, iteration_count

if __name__ == '__main__':
    def random_symmetric_matrix(m):
        """
        Generate a symmetric random mxm matrix.

        :param m: the desired number of rows and columns
        :type m: integer
        """
        A = nr.rand(m, m)
        return (A + A.T)/2
    
    def random_tridiagonal_matrix(m):
        return sl.hessenberg(random_symmetric_matrix(m))
    
    def test_count_eig_negative(n_iterations, m):
        for i in range(n_iterations):
            A = random_tridiagonal_matrix(m)
            x = nr.random()*2 - 1
            our = count_eig_negative(A, x)
            sls = len([lamda for lamda in sl.eig(A)[0] if lamda <= x])  
            if our != sls:
                print("test_count_eig_negative failed")
                print(A)
                print(x)
                print(sl.eig(A)[0])
                print(our, sls)
                sys.exit(1)
    test_count_eig_negative(100, 10)
    