import numpy as np

vp = np.array( [ [0,0,0,1,1,2]
               , [1,2,3,2,3,3] ], dtype=np.int64 )
vs = np.array( [ [0,0,1]
               , [1,2,2] ], dtype=np.int64 )
vl = np.array( [ [0]
               , [1] ], dtype=np.int64 )

c  = 299792458.0
u0 = 4e-7*np.pi
e0 = 1/(u0*c**2)

def ntet(p): # p.shape: (4, 3)
  q = p[1:] - p[0] #; print(q)
  n = np.empty_like(p, dtype=np.float64)
  n[1:] = np.cross(q[[1, 2, 0]], q[[2, 0, 1]])
  n[0] = -n[1:].sum(axis=0) #; print(n)
  vol = np.inner(q[0], n[1])
  n /= vol
  return n, vol

def make_stiff(n, vol): # n.shape: (4, 3)
  pr = np.cross(n[vp[0]], n[vp[1]])
  ip = np.inner(pr, pr)
  ip *= 4/6*np.abs(vol)
  return ip

def make_mass_core(v, n, vol): # n.shape: (4, 3)
  cc = (v[[1,1,0]][..., np.newaxis, :] == v[[1,0,0]][..., np.newaxis]) + 1
  ma  = np.inner(n[v[0]], n[v[0]])*cc[0]
  m1  = np.inner(n[v[1]], n[v[0]])*cc[1]
  ma -= m1
  ma -= m1.T
  ma += np.inner(n[v[1]], n[v[1]])*cc[2]
  ma *= np.abs(vol)
  return ma

def make_mass(n, vol): # n.shape: (4, 3)
  return make_mass_core(vp, n, vol)/120

def bound(n, vol): # n.shape: (4, 3)
  return make_mass_core(vs, n, vol)/24

def ntri2(p): # p: (3, 3)
  p[1:] -= p[0]
  p[0] = np.cross(p[1], p[2])
  jacob = (p[0]**2).sum(axis=-1)
  n = np.empty_like(p, dtype=np.float64)
  n[1:] = np.cross(p[[2, 0]], p[[0, 1]])
  n[0] = -n[1]-n[2]
  return n/jacob, np.sqrt(jacob)

def volume(freq, p1, sigma, epsilon, mu):
  n, vol = ntet(p1)
  stiff = make_stiff(n, vol)
  mass = make_mass(n, vol)
  w = 2*np.pi*freq
  return stiff/mu - w*(w*epsilon-1j*sigma)*mass

def absorb(freq, p1):
  n, area = ntri2(p1)
  ab = bound(n, area)
  return 2j*np.pi*freq*np.sqrt(e0/u0)*ab

def isrc1(v, nfn, freq, p1, i_density):
  n, jacob = nfn(p1)
  x  = n[v[1]] - n[v[0]]
  x *= np.array(i_density)
  x  = x.sum(axis=-1)
  x *= np.abs(jacob)
  x = -2j*np.pi*freq*x
  return x

def isrc(freq, p1, i_density):
  x = isrc1(vl, lambda p: (p[[1, 0]] - p[[0, 1]], 1.0), freq, p1, i_density)
  return x/2

def isrc_2d(freq, p1, i_density):
  x = isrc1(vs, ntri2, freq, p1, i_density)
  return x/6

def isrc_3d(freq, p1, i_density):
  x = isrc1(vp, ntet, freq, p1, i_density)
  return x/24
