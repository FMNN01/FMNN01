#!/usr/bin/env python3
# encoding: utf-8
#
# Authors:
#  Karl Lindén <karl.linden.887@student.lu.se>
#  Oscar Nilsson <erik-oscar-nilsson@live.se>
#

import numpy as np
import scipy.linalg as sl

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

def apply_qr_factorization():
    Q, R = sl.qr(T)

    # Make R a square matrix.
    n = R.shape[1]
    R = R[:n,:]

    # Dismiss the latter entries of np.dot(Q.T, v).
    b = np.dot(Q.T, v)[:n]

    # Perform backwards substitution.
    return backwards_substitution(R, b)

def apply_svd():
    pass

# FIXME: do something w/ this
ahat1 = solve_normal_equations()
ahat2 = apply_qr_factorization()
ahat3 = apply_svd()
print(ahat1)
print(ahat2)

