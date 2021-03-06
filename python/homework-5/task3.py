#!/usr/bin/env python3
# encoding: utf-8

# System imports.
import sys

# Numerical analysis imports.
import numpy as np
import numpy.random as nr
import scipy.linalg as sl

def count_eig_negative(A, x):
    """
    We compute the sub-matrices of A - xI, to compute the number of
    negative eigenvalues by counting the number of sign changes.

    :param A: a square tridiagonal matrix
    :type A: np.ndarray
    :param x: shift variable
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

    # Deflate the problem is possible, otherwise start counting now.
    try:
        j = isclose.index(True)
        return count_eig_between(A[:j+1, :j+1], a, b) + \
               count_eig_between(A[j+1:, j+1:], a, b)
    except ValueError:
        # Count the eigenvalues less than or equal to b, then subtract
        # the count of the interval less than a, i.e. (-infty, a].
        return count_eig_negative(A, b) - count_eig_negative(A, a)

def find_eig_between(A, a, b, atol=1.e-8):
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
        return find_eig_between(A, a, mid, atol=atol) + \
               find_eig_between(A, mid, b, atol=atol)

if __name__ == '__main__':
    def random_symmetric_matrix(m):
        """
        Generate a symmetric random mxm matrix.

        :param m: the desired number of rows and columns
        :type m: int
        """
        A = nr.rand(m, m)
        return (A + A.T)/2

    def random_tridiagonal_matrix(m):
        """
        Generate a symetric trididagonal matrix.

        :param m: the desired dimension of the square matri x.
        :type m: int 
        """
        return sl.hessenberg(random_symmetric_matrix(m))

    def test_count_eig_negative(n, m):
        """
        Test function for count_eig_negative.

        :param n: number of random matrices to test
        :type n: integer
        :param m: size of the matrices to test
        :type m: int
        """
        for i in range(n):
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

    def test_count_eig_between(n, m):
        """
        Test function for count_eig_between.

        :param n: number of random matrices to test
        :type n: integer
        :param m: size of the matrices to test
        :type m: int
        """
        for i in range(n):
            A = random_tridiagonal_matrix(m)

            # Randomize the interval.
            endpoints = nr.random(2)
            a = min(endpoints)
            b = max(endpoints)

            our = count_eig_between(A, a, b)
            sls = len(
                    [lamda for lamda in sl.eig(A)[0] if a < lamda <= b])
            if our != sls:
                print("test_count_eig_between failed")
                print(A)
                print(a)
                print(b)
                print(sl.eig(A)[0])
                print(our, sls)
                sys.exit(1)

    def test_find_eig_between(n, m):
        """
        Test function for find_eig_between.

        :param n: number of random matrices to test
        :type n: integer
        :param m: size of the matrices to test
        :type m: int
        """
        for i in range(n):
            A = random_tridiagonal_matrix(m)

            # Randomize the interval.
            endpoints = nr.random(2)
            a = min(endpoints)
            b = max(endpoints)

            x = nr.random()*2 - 1
            our = np.array(find_eig_between(A, a, b))
            sls = [lamda for lamda in sl.eig(A)[0] if a < lamda <= b]

            # The eigenvalues should be compared, so sort them to make
            # the test independent of the returned order.
            our.sort()
            sls.sort()
            if not np.isclose(sls, our).all():
                print("test_find_eig_between failed")
                print(A)
                print(a)
                print(b)
                print(our, sls)
                sys.exit(1)

    # Test all functions.
    N = 300
    M = 30
    test_count_eig_negative(N, M)
    test_count_eig_between(N, M)
    test_find_eig_between(N, M)
