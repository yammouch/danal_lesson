import numpy as np

def kin(k_basis, lc):
    k = 2*np.pi/np.array(lc)[None, :]*k_basis
    x = k[:, None, :] - k[None, :, :]
    x = x**2
    x = x.sum(axis=-1)
    x = -0.5*x
    return x

def symmetrize(a):
    for i in range(a.ndim):
        l = a.shape[i]
        if l % 2 == 0:
            a[(slice(None),)*i + (l//2,)] *= 0.5
    return a
