#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import scipy.linalg as sl
import matplotlib.pylab as plt

M = np.array([[ 5  , 0,  0, -1],
              [ 1  , 0, -2,  1],
              [-1.5, 1, -2,  1],
              [-1  , 3,  1, -3]])
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

# We want to draw circles so keep the aspect ratio equal.
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

# Draw all Gerschgorin circles and compute sensible limits for the axes.
xmin = float("inf")
xmax = float("-inf")
ymin = float("inf")
ymax = float("-inf")
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

    # Update the axes limits so that this circle fits.
    xmin = min(xmin, center[0] - radius)
    xmax = max(xmax, center[0] + radius)
    ymin = min(ymin, center[1] - radius)
    ymax = max(ymax, center[1] + radius)

# Set the limits of the axes.
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Add a grid and save the figure.
plt.grid()
plt.savefig('task4.eps')
