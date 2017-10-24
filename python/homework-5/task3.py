# encoding: utf-8

# System imports.
import sys

# Numerical analysis imports.
import numpy as np
import numpy.random as nr
import scipy.linalg as sl

def count_eig_negative(A, x):
    """
    We compute the sub matrices of A, to compute
    the number of negative eigenvalues by counting the
    number of sign changes.
    :param A: a square matrix
    :type A: np.ndarray
    :param x: evaluation point for p(x)
    :type x: float
    """

    m = A.shape[0]

    # Expanding the minor determinants recursively.
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
    Count the matrix eigenvalues that lie in the interval (a,b].

    :param A: the matrix whose eigenvalues to count
    :type A: np.ndarray
    :param a: the lower limit of the interval
    :type a: float
    :param b: the upper limit of the interval
    :type b: float
    """
    # Deflate the problem if possible.
    # Otherwise just invoke count_eig_negative.
    m = A.shape[0]
    # Handle the trivial case of a 1x1 matrix.
    if m == 1:
        return [A[0, 0]], 0
    # Boolean array representing the superdiagonal elements.
    super_diagonal = np.tri(m, k=1, dtype=bool) - np.tri(m, dtype=bool)
    isclose = list(np.isclose(A[super_diagonal], 0))
    j = isclose.index(True)
    if j > 0:
        return count_eig_between(A[:j+1, :j+1], a, b) + \
               count_eig_between(A[j+1:, j+1:], a, b)
    # Here we remember that we count the number of eigenvalues less than a,
    # then we take away the interval less than b, i.e. (b, A].
    return count_eig_negative(A, b) - count_eig_negative(A, a)
def find_eig_between(A, a, b, atol = 1.e-8):
    """
    Count the matrix eigenvalues that lie in the interval (a,b].

    :param A: the matrix whose eigenvalues to count
    :type A: np.ndarray
    :param a: the lower limit of the interval
    :type a: float
    :param b: the upper limit of the interval
    :type b: float
    :param atol:
    :type: float
    """
    # We start by finding the midpoint. Then we see how many
    # eigenvalues there are in these interval.
    mid = (a+b)/2
    if count_eig_between(A, a, b) == 0:
        return []
    elif mid - a <= atol:
        return [mid]
    else:
        return find_eig_between(A, a, mid, atol = atol) + \
               find_eig_between(A, a, mid, atol = atol)

if __name__ == '__main__<':
    def random_symmetric_matrix(m):
        """
        Generate a symmetric random mxm matrix.

        :param m: the desired number of rows and columns
        :type m: integer
        """
        A = nr.rand(m, m)
        return (A + A.T)/2

    def random_tridiagonal_matrix(m):
        """
        Generate a symetric trididagonal matrix
        :param m: the desired dimension of the square matri x.
        :type m: integer
        """
        return sl.hessenberg(random_symmetric_matrix(m))

    def test_count_eig_negative(n_many_tests, m):
        """
        Test method for the count_eig_negative.
        :param n_many_times: number of random matrices we test.
        :type n_many_times: integer
        :param m: size of the matrices we are testing.
        :type m: integer
        """
        for i in range(n_many_tests):
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
