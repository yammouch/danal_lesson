import numpy as np

vp = np.array( [ [0,0,0,1,1,2]
               , [1,2,3,2,3,3] ], dtype=np.int64 )

def ntet(p): # p.shape: (4, 3)
  q = p[1:] - p[0] #; print(q)
  n = np.empty_like(p, dtype=np.float64)
  n[1:] \
  = q[[[1],[2],[0]], [1,2,0]] * q[[[2],[0],[1]], [2,0,1]] \
  - q[[[1],[2],[0]], [2,0,1]] * q[[[2],[0],[1]], [1,2,0]]
  n[0] = -n[1:].sum(axis=0) #; print(n)
  vol = (q[0]*n[1]).sum(axis=-1)
  n /= vol
  return n, vol

def make_stiff(n, vol): # n.shape: (4, 3)
  pr = n[vp[0][:, np.newaxis], [1,2,0]] * n[vp[1][:, np.newaxis], [2,0,1]] \
     - n[vp[0][:, np.newaxis], [2,0,1]] * n[vp[1][:, np.newaxis], [1,2,0]]
  ip = (pr * pr[:, np.newaxis]).sum(axis=-1)
  ip *= 4/6*np.abs(vol)
  return ip

def make_mass(n, vol): # n.shape: (4, 3)
  cc = (vp[[1,1,0]][..., np.newaxis, :] == vp[[1,0,0]][..., np.newaxis]) + 1
  ma  = (n[vp[0]]*n[vp[0]][..., np.newaxis, :]).sum(axis=-1)*cc[0]
  m1  = (n[vp[0]]*n[vp[1]][..., np.newaxis, :]).sum(axis=-1)*cc[1]
  ma -= m1
  ma -= m1.T
  ma += (n[vp[1]]*n[vp[1]][..., np.newaxis, :]).sum(axis=-1)*cc[2]
  ma *= np.abs(vol)/120
  return ma

def ntri2(p): # p: (..., 3, 3)
  p[..., 1:, :] -= p[..., [0], :]
  p[..., 0, :] = p[..., 1, [1, 2, 0]]*p[..., 2, [2, 0, 1]] \
               - p[..., 1, [2, 0, 1]]*p[..., 2, [1, 2, 0]]
  jacob = (p[..., 0, :]**2).sum(axis=-1)
  n = np.empty_like(p, dtype=np.float64)
  n[..., 1, :] = p[..., 2, [1, 2, 0]]*p[..., 0, [2, 0, 1]] \
               - p[..., 2, [2, 0, 1]]*p[..., 0, [1, 2, 0]]
  n[..., 2, :] = p[..., 0, [1, 2, 0]]*p[..., 1, [2, 0, 1]] \
               - p[..., 0, [2, 0, 1]]*p[..., 1, [1, 2, 0]]
  n[..., 0, :] = -n[..., 1, :]-n[..., 2, :]
  return n/jacob[..., None, None], np.sqrt(jacob)

def bound(n, area): # n: (..., 3, 3), area: (...)
  a0 = (n[..., [[0],[0],[1]], :] * n[..., [[0,0,1]], :]).sum(axis=-1)
  a0[..., [0, 1, 1, 2, 2], [0, 1, 2, 1, 2]] *= 2
  a1 = (n[..., [[0],[0],[1]], :] * n[..., [[1,2,2]], :]).sum(axis=-1)
  a1[..., 0, 2] *= 2
  a2 = (n[..., [[1],[2],[2]], :] * n[..., [[0,0,1]], :]).sum(axis=-1)
  a2[..., 2, 0] *= 2
  a3 = (n[..., [[1],[2],[2]], :] * n[..., [[1,2,2]], :]).sum(axis=-1)
  a3[..., [0, 0, 1, 1, 2], [0, 1, 0, 1, 2]] *= 2
  return (a0 - a1 - a2 + a3)*(area[..., None, None]/24)
