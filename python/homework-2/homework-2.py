# Authors:
#  Karl Lindén <karl.linden.887@student.lu.se>
#  Oscar Nilsson <erik-oscar-nilsson@live.se>

import numpy as np
import numpy.linalg as nl
import numpy.random as nr

# Answer to "What is meant by ”Gram-Schmidt is unstable”?":
# Run this program. It will output that Gram-Schmidt performs badly with
# almost all the random matrices it is fed with.

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

    def householder(self):
        """
        QR factorize the matrix given to :meth:`__init__` using
        Householder reflections.
        """
        # The algorithm successively computes Q and R. These are the
        # start values. Note that np.copy is needed to not overwrite the
        # A matrix.
        Q = np.eye(self.m)
        R = np.copy(self.A)

        for j in range(self.n):
            # Find a unit vector that makes only the first entry of the
            # first column of the jj submatrix non-zero.
            B = R[j:,j:]
            vhat = self._householder(B)

            # Reflect the lower part of the remaining R matrix. Start at
            # column j, since the first j-1 columns all have zeroes
            # in the lower part. This could have been a matrix
            # multiplication, but iteration over the columns is faster.
            for k in range(j, self.n):
                R[j:,k] -= 2 * vhat * np.inner(vhat, R[j:,k])

            # Reflect the right part of the Q matrix. This is the same
            # as multiplying together Q with (I-vv^T) from the right,
            # but this iteration is faster. Here
            #   v = (0, 0, ..., 0, vhat]
            # is the m-vector which is vhat, but with leading zeroes.
            # (vhat is an m-j vector).
            for i in range(self.m):
                Q[i,j:] -= 2 * np.inner(Q[i,j:], vhat) * vhat

        return Q, R

    def _householder(self, A):
        """
        Find a unit vector such that when the first column of A is
        reflected along v the result is a (column) vector whose only
        non-zero entry is the first one.
        """
        m = A.shape[0]
        a = A[:,0]
        ahat = np.array([nl.norm(a, 2)] + (m-1)*[0])
        v = ahat - a
        if not np.allclose(v, 0):
            v /= nl.norm(v, 2)
        return v

    def givens(self):
        """
        QR factorize the matrix given to :meth:`__init__` by Givens
        rotations.
        """
        # "Inspiration" for this algorithm comes from:
        #   https://en.wikipedia.org/wiki/Givens_rotation

        Q = np.eye(self.m)
        R = np.copy(self.A)

        # Make entries in R successively zero. Iterate through the
        # columns begin introducing zeros from the last row up until the
        # main diagonal.
        for i in range(self.n):
            for k in range(self.m - 1, i, -1):
                # We want to make eliminate entry k,i so construct a
                # Givens rotate rows i and k so that the desired entry
                # is zero.
                a = R[i,i]
                b = R[k,i]
                G = np.array([[ a, b],
                              [-b, a]]) / np.hypot(a, b)

                # Only operate on the necessary parts of the matrices,
                # multiplying with a full blown Givens rotation gives
                # extremely poor performance. The Givens rotation matrix
                # is mostly an identity matrix, so this is possible.
                R[(i,k),:] = np.dot(G, R[(i,k),:])
                Q[:,(i,k)] = np.dot(Q[:,(i,k)], G.T)

        return Q, R

# A list of implement QR factorization methods.
qr_factorization_methods = [
    "numpy",
    "householder",
    "givens"
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

def test_qr_factorization_results(A, Q, R):
    """
    Summarize the result of a QR factorization.

    :param A: the original matrix
    :type A: np.ndarray
    :param Q: the orthogonal matrix
    :type Q: np.ndarray
    :param R: the upper rectangular matrix
    :type R: np.ndarray
    """
    test_result("A = QR", np.dot(Q, R), A)

def test_orthogonalization_results(Q):
    """
    Summarize the result of an orthogonalization.

    :param Q: the result of the orthogonalization
    :type Q: np.ndarray
    """
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

def test_qr_factorization(method_name, A):
    """
    Test a QR factorization method of the QRFactorization class.

    :param method_name: the name of the method to test, e.g.
    "gramschmidt"
    :type method_name: str
    :param A: the matrix to test
    :type A: numpy.ndarray
    """
    # Be pollite and print what we are doing.
    print("QR factorizing random ({},{}) matrix with {}:".format(
        A.shape[0], A.shape[1], method_name))

    # Perform QR factorization.
    qrf = QRFactorization(A)
    method = getattr(qrf, method_name)
    Q, R = method()

    test_qr_factorization_results(A, Q, R)
    test_orthogonalization_results(Q)

def test_orthogonalization(method_name, A):
    """
    Test an orthogonalization method of the Orthogonalization class.

    :param method_name: the name of the method to test, e.g.
    "gramschmidt"
    :type method_name: str
    :param A: the matrix to test
    :type A: numpy.ndarray
    """
    # Be pollite and print what we are doing.
    print("Orthogonalizing random ({},{}) matrix with {}:".format(
        A.shape[0], A.shape[1], method_name))

    # Compute orthogonalization.
    orth = Orthogonalization(A)
    method = getattr(orth, method_name)
    Q = method()

    test_orthogonalization_results(Q)

# List of orthogonalization methods to test.
orthogonalization_methods = ["gramschmidt"]

BASE = 2
PMAX = 11

# Test the algorithms with quadratic (diff = 0) matrices and with
# non-quadratic (diff = 4) matrices.
for diff in [0, 4]:
    for p in range(PMAX):
        n = pow(BASE, p)
        m = n + diff
        A = random_bad_matrix(m, n)
        for method_name in orthogonalization_methods:
            test_orthogonalization(method_name, A)
        for method_name in qr_factorization_methods:
            test_qr_factorization(method_name, A)
