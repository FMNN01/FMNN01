import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl

u = np.array([2,3,5,7,11,])
v = np.array([3,1,4])
A = np.outer(u, v)

U, s, Vh = sl.svd(A)

def norm_and_unit(w):
    norm = nl.norm(w, 2)
    return [norm, w/norm]
u_norm, u_unit = norm_and_unit(u)
v_norm, v_unit = norm_and_unit(v)

print("u_unit:", u_unit)
print("v_unit:", v_unit)
print("u_norm * v_norm:", u_norm * v_norm)
print("A", A)
print("U", U)
print("sigma:", np.diag(s))
print("Vh:", Vh)
