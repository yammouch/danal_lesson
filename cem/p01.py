import numpy as np

vp = np.array( [ [0,0,0,1,1,2]
               , [1,2,3,2,3,3] ], dtype=np.int64 )

vs = np.array( [ [0,0,1]
               , [1,2,2] ], dtype=np.int64 )

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

def make_mass_core(v, n, vol): # n.shape: (4, 3)
  cc = (v[[1,1,0]][..., np.newaxis, :] == v[[1,0,0]][..., np.newaxis]) + 1
  ma  = (n[v[0]]*n[v[0]][..., np.newaxis, :]).sum(axis=-1)*cc[0]
  m1  = (n[v[0]]*n[v[1]][..., np.newaxis, :]).sum(axis=-1)*cc[1]
  ma -= m1
  ma -= m1.T
  ma += (n[v[1]]*n[v[1]][..., np.newaxis, :]).sum(axis=-1)*cc[2]
  ma *= np.abs(vol)
  return ma

def make_mass(n, vol): # n.shape: (4, 3)
  return make_mass_core(vp, n, vol)/120

def bound(n, vol): # n.shape: (4, 3)
  return make_mass_core(vs, n, vol)/24

def ntri2(p): # p: (3, 3)
  p[1:] -= p[0]
  p[0] = p[1, [1, 2, 0]]*p[2, [2, 0, 1]] \
       - p[1, [2, 0, 1]]*p[2, [1, 2, 0]]
  jacob = (p[0]**2).sum(axis=-1)
  n = np.empty_like(p, dtype=np.float64)
  n[1] = p[2, [1, 2, 0]]*p[0, [2, 0, 1]] \
       - p[2, [2, 0, 1]]*p[0, [1, 2, 0]]
  n[2] = p[0, [1, 2, 0]]*p[1, [2, 0, 1]] \
       - p[0, [2, 0, 1]]*p[1, [1, 2, 0]]
  n[0] = -n[1]-n[2]
  return n/jacob, np.sqrt(jacob)
