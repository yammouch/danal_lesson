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

class volume(object):

  def __init__(self, sigma, epsilon, mu):
    super().__init__()
    self.sigma   = sigma
    self.epsilon = epsilon
    self.mu      = mu

  def __call__(self, freq, p1):
    n, vol = ntet(p1)
    stiff = make_stiff(n, vol)
    mass = make_mass(n, vol)
    w = 2*np.pi*freq
    return stiff/self.mu - w*(w*self.epsilon-1j*self.sigma)*mass

def absorb(freq, p1):
  n, area = ntri2(p1)
  ab = bound(n, area)
  return 2j*np.pi*freq*np.sqrt(e0/u0)*ab

class isrc(object):

  v_table   = { 1: vl, 2: vs, 3: vp }
  nfn_table = { 1: lambda p: (p[[1, 0]] - p[[0, 1]], 1.0)
              , 2: ntri2, 3: ntet }
  den_table = { 1: 2, 2: 6, 3: 24}

  def __init__(self, dim, i_density):
    self.i_density = np.array(i_density)
    self.v   = self.v_table[dim]
    self.nfn = self.nfn_table[dim]
    self.den = self.den_table[dim]

  def __call__(self, freq, p1):
    n, jacob = self.nfn(p1)
    x  = n[self.v[1]] - n[self.v[0]]
    x *= np.array(self.i_density)
    x  = x.sum(axis=-1)
    x *= np.abs(jacob)
    x = -2j*np.pi*freq*x
    return x/self.den
