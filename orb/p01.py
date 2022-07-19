import numpy as np

def kin(k_basis, lc):
  diff = k_basis[:, None, :] - k_basis[None, :, :]
  diff2 = diff**2
  diff2sum = diff2.sum(axis=-1)
  return -2*(np.pi/lc)**2*diff2sum
