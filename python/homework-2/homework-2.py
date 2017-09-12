import numpy as np
import numpy.linalg as nl
import numpy.random as nr

class QRFactorization:
    """
    QR factorize an (m,n) matrix in different ways.
    """

    def __init__(self, A):
        """
        Class initializer.

        :param A: the matrix to QR factorize
        :type A: np.ndarray or castable to such
        """
        if not isinstance(A, np.ndarray):
            A = np.array(A)
        self.A = A
        self.m, self.n = A.shape

    def numpy(self):
        """
        QR factorize the matrix given to :meth:`__init__` using the QR
        factorization algorithm supplied by NumPy.
        """
        return nl.qr(self.A)

# A list of implement QR factorization methods.
qr_factorization_methods = [
    "numpy"
]

class QROrthogonalizationWrapper:
    """
    An orthogonalization wrapper around the QR factorization class.
    """

    def __init__(self, qr_factorization, name):
        """
        Class initializer.

        :param qr_factorization: the instance of QRFactorization to use
        :type qr_factorization: QRFactorization
        :param name: the name of the QR factorization method to wrap
        :type name: string
        """
        self.method = getattr(qr_factorization, name)

    def __call__(self):
        """
        Call this wrapper.
        """
        return self.method()[0]

class Orthogonalization:
    """
    Orthogonalize an (m,n) in different ways.
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

        # Orthogonalization is a part of QR factorization so create
        # wrappers for the implemented QR factorization methods.
        qr_factorization = QRFactorization(A)
        for method in qr_factorization_methods:
            wrapper = QROrthogonalizationWrapper(
                    qr_factorization, method)
            setattr(self, method, wrapper)

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

# Width of the first column when pretty-printing.
WIDTH = 23

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
        print("  {}: not applicable".format("Determinant".rjust(WIDTH)))

# List of method names to test.
orthogonalization_methods = ["gramschmidt"] + qr_factorization_methods

BASE = 2
PMAX = 11

# Test the orthogonalization algorithms with quadratic (diff = 0)
# matrices and with non-quadratic (diff = 4) matrices.
for diff in [0, 4]:
    for p in range(PMAX):
        n = pow(BASE, p)
        m = n + diff
        A = random_bad_matrix(m, n)
        for method_name in orthogonalization_methods:
            test(method_name, A)
