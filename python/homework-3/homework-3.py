#!/usr/bin/env python3
# encoding: utf-8
#
# Authors:
#  Karl Lindén <karl.linden.887@student.lu.se>
#  Oscar Nilsson <erik-oscar-nilsson@live.se>
#

import numpy as np

DATA_FILENAME = "data.txt"

# Read the data file. This is not the code from the homepage, since
# interactivity was not needed.
with open(DATA_FILENAME) as f:
    lines = [l.strip().split(',') for l in f.readlines()]
    A = [[float(l[0]), float(l[1])] for l in lines]
    A = np.array(A)
    t = A[:,0]
    v = A[:,1]

print(t)
print(v)
