import numpy as np
import numpy.linalg as nl
import numpy.random as nr

# Width of the first column when pretty-printing.
WIDTH = 23

class Orthogonalization:
    """
    Orthogonalize an (m,n), where m >= n, in different ways.
    """

    def __init__(self, A):
        """
        Class initializer.

        :param A: the matrix to orthogonalize
        :type A: np.ndarray or castable to such
        """
        if not isinstance(A, np.ndarray):
            A = np.array(A)
        self.A = A
        self.m, self.n = A.shape

    def gramschmidt(self):
        """
        Orthogonalize the matrix given to :meth:`__init__` using the
        Gram-Schmidt algorithm.

        Linearly dependent columns of A will be removed, so the returned
        matrix is not guaranteed to have the same dimensions as the one
        given.
        """
        # Projection matrix.
        P = np.eye(self.m)

        # Columns of the resulting matrix Q. To be filled in by the
        # algorithm.
        Q_columns = []

        # Iterate through the columns and project each column.
        for j in range(self.n):
            # Extract column and project it.
            a = self.A[:,j]
            q = np.dot(P, a)

            # If the projection is zero then the column must be
            # dismissed, i.e. nothing shall be done. Otherwise append
            # the normalized vector to the list and update the
            # projection matrix.
            if not np.allclose(q, 0):
                q /= nl.norm(q)
                Q_columns.append(q)
                P -= np.outer(q, q)

        return np.column_stack(Q_columns)

    def qr(self):
        """
        Orthogonalize the matrix given to :meth:`__init__` using the QR
        factorization algorithm supplied by NumPy.
        """
        return nl.qr(self.A)[0]

def randscale():
    """
    Return a random scale factor that is possibly very big or very
    small for extra numerical instability.
    """
    return pow(10, nr.random() * 6 - 3)

def random_bad_matrix(m, n):
    """
    Return a matrix that will most often yield numerical instabilities.
    """
    A = nr.rand(m, n)
    for i in range(m):
        A[i,:] *= randscale()
    for j in range(n):
        A[:,j] *= randscale()
    return A

def test_result(message, value, expected):
    """
    Helper function for printing a test result.
    """
    # Give this test a score of good or bad.
    score = np.allclose(value, expected) and "good" or "bad"

    # Compute the difference from the actual value and the expected
    # value. If the difference is a vector take the two norm of it.
    diff = abs(value - expected)
    if diff.shape:
        diff = nl.norm(diff, 2)

    # Print the result.
    print("  {}: {} {}".format(message.rjust(WIDTH), score, diff))

def test(method_name, A):
    """
    Tests an orthogonalization method of the Orthogonalization class by
    feeding it with a random matrix of a given shape.

    :param method_name: the name of the method to test, e.g.
    "gramschmidt"
    :type method_name: str
    :param A: the matrix to test
    :type A: numpy.ndarray
    """
    # Compute orthogonalization.
    orth = Orthogonalization(A)
    method = getattr(orth, method_name)
    Q = method()

    print("Random ({},{}) matrix with {}:".format(
        A.shape[0], A.shape[1], method_name))

    test_result("2-norm", nl.norm(Q, 2), 1)

    QTQ = np.dot(Q.T, Q)
    I = np.eye(*QTQ.shape)
    deviation = nl.norm(I - QTQ, 2)
    test_result("Deviation from identity", deviation, 0)

    eigenvalues = nl.eigh(QTQ)[0]
    test_result("Eigenvalues", eigenvalues, 1)

    if Q.shape[0] == Q.shape[1]:
        test_result("Determinant", abs(nl.det(Q)), 1)
    else:
        print("  {}: not applicable".format("Determinant"))

# List of method names to test.
method_names = ["gramschmidt", "qr"]

BASE = 2
PMAX = 11
DIFF = 4

# Test quadratic matrices.
for p in range(PMAX):
    m = pow(BASE, p)
    A = random_bad_matrix(m, m)
    for method_name in method_names:
        test(method_name, A)

# Test non-quadratic matrices.
for p in range(PMAX):
    n = pow(BASE, p)
    m = n + DIFF
    A = random_bad_matrix(m, n)
    for method_name in method_names:
        test(method_name, A)
