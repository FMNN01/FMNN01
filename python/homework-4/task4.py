#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import scipy.linalg as sl
import matplotlib.pylab as plt

M = np.array([[ 5  , 0,  0, -1],
              [ 1  , 0, -3,  1],
              [-1.5, 1, -2,  1],
              [-1  , 5,  3, -3]])
m = M.shape[0]

def A(p):
    """
    Compute the matrix A(p).

    :param p: the parameter
    :type p: float
    """
    diag = np.eye(m, dtype=bool)
    off_diag = np.logical_not(diag)
    N = np.copy(M)
    N[off_diag] *= p
    return N

def get_color(i):
    """
    Return the i:th color.
    """
    colors = ['red', 'blue', 'green', 'purple']
    return colors[i % len(colors)]

# Save the eigenvalues as rows in a matrix called E. Consequently each
# column has the eigenvalues originating from a given center of a
# Gerschgorin disk.
P = np.linspace(0, 1, 1000)
eigenvalues = [sl.eig(A(p))[0] for p in P]
E = np.vstack(eigenvalues)

# Plot each eigenvalue trajectory, by extracting the real and imaginary
# part of each column of E.
for i in range(m):
    X = E[:,i].real
    Y = E[:,i].imag
    c = get_color(i)
    plt.plot(X, Y, color=c)
    plt.plot(X[0], Y[0], 'bo', color=c)
    plt.plot(X[-1], Y[-1], 'bo', color=c)


# Modify the axes to give a good picture. Also ax is needed in the
# circle creating for loop below.
ax = plt.gca()
ax.set_xlim(-13, 7)
ax.set_ylim(-10, 10)
ax.set_aspect('equal', adjustable='box')

# Draw all Gerschgorin circles.
for i in range(m):
    # The center is the element on the diagonal.
    center = (M[i,i].real, M[i,i].imag)

    # The radius is the sum of the absolute value of the off-diagonal
    # elements.
    off_diag_on_row = np.ones(m, dtype=bool)
    off_diag_on_row[i] = False
    radius = sum(abs(M[i,:][off_diag_on_row]))

    # Draw the circle.
    circle = plt.Circle(
            (M[i,i],0),
            radius,
            color=get_color(i),
            fill=False)
    ax.add_artist(circle)

# Add a grid and save the figure.
plt.grid()
plt.savefig('task4.eps')
