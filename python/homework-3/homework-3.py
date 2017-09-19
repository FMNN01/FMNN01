#!/usr/bin/env python3
# encoding: utf-8
#
# Authors:
#  Karl Lindén <karl.linden.887@student.lu.se>
#  Oscar Nilsson <erik-oscar-nilsson@live.se>
#

import time

import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
import matplotlib.pylab as plt

########################################################################
## TASK 1 ##############################################################
########################################################################

DATA_FILENAME = "data.txt"

# Read the data file. This is not the code from the homepage, since
# interactivity was not needed.
with open(DATA_FILENAME) as f:
    lines = [l.strip().split(',') for l in f.readlines()]
    A = [[float(l[0]), float(l[1])] for l in lines]
    A = np.array(A)
    t = A[:,0]
    v = A[:,1]

T = np.vander(t, 5, increasing=True)

# A list containing methods that implement solving the least squares
# problem.
lstsq_impls = []

def lstsq_impl(func):
    lstsq_impls.append(func)

@lstsq_impl
def solve_normal_equations():
    A = np.dot(T.T, T)
    b = np.dot(T.T, v)
    return sl.solve(A, b)

def backwards_substitution(R, b):
    m = R.shape[0]
    rmm = R[m-1,m-1]
    r = R[:m-1,m-1] / rmm

    am = b[m-1] / rmm

    if m > 1:
        bprime = b[:m-1] - r*b[m-1]
        Rprime = R[:m-1,:m-1]
        aprime = backwards_substitution(Rprime, bprime)

        a = np.array(list(aprime) + [0])
        a[m-1] = am
        return a
    else:
        return np.array([am])

@lstsq_impl
def apply_qr_factorization():
    Q, R = sl.qr(T)

    # Make R a square matrix.
    n = R.shape[1]
    R = R[:n,:]

    # Dismiss the latter entries of np.dot(Q.T, v).
    b = np.dot(Q.T, v)[:n]

    # Perform backwards substitution.
    return backwards_substitution(R, b)

@lstsq_impl
def apply_svd():
    n = T.shape[1]
    U, s, Vh = sl.svd(T)
    b = np.dot(U.T, v)[:n] / s
    return np.dot(Vh.T, b)

@lstsq_impl
def apply_numpy_lstsq():
    return nl.lstsq(T, v)[0]

for func in lstsq_impls:
    # Print name of this function, the result and the time it took.
    clk1 = time.clock()
    a = func()
    clk2 = time.clock()
    print("Function:", func.__name__)
    print("  Result:", a)
    print("    Time:", clk2 - clk1)
    print()

    # Convert the a vector to coeffecients and create a linspace for the
    # t-axis.
    c = list(a)
    c.reverse()
    X = np.linspace(min(t), max(t), 1000)

    # Plot the data and the polynomial fit.
    dataplot, = plt.plot(t, v, 'o')
    dataplot.set_label('data')
    polyplot, = plt.plot(X, np.polyval(c, X))
    polyplot.set_label('p(x)')
    plt.title(func.__name__)
    plt.xlabel('t')
    plt.ylabel('v')
    plt.legend()
    plt.grid()
    plt.savefig(func.__name__ + ".png")
    plt.close()

########################################################################
## TASK 2 ##############################################################
########################################################################


