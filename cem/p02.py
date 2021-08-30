import numpy as np
import p01

def solve(vrt, spt, edg, freq):
  u0 = 4e-7*np.pi
  e0 = 8.854e-12
  b = np.zeros((6, 6), dtype=np.complex128)
  for p, e in zip(spt, edg):
    n, area = p01.ntri(vrt[:, p])
    b1 = p01.bound(n, area)
    b[e[:,None],e[None,:]] += 2j*np.pi*freq*np.sqrt(u0*e0)*b1
  n, vol = p01.ntet(vrt)
  stiff = p01.make_stiff(n, vol)
  mass = p01.make_mass(n, vol)
  lhs = stiff/(4e-7*np.pi) - (2*np.pi*freq)**2*e0*mass + b
  rhs = np.zeros(6, dtype=np.complex128)
  rhs[2] = 2j*np.pi*freq*2*np.sqrt(3)
  lhs[0:2, :] = 0
  lhs[3:6, :] = 0
  lhs[[0,1  ], [0,1  ]] = 1
  lhs[[3,4,5], [3,4,5]] = 1
  #print(lhs)
  #print(rhs)
  sol = np.linalg.solve(lhs, rhs)
  return sol

if __name__ == '__main__':
  vrt = np.array \
  ( [ [ 0, 1, 0, 0    ]
    , [ 0, 0, 1, 0    ]
    , [ 0, 0, 0, 1e-3 ] ] )
  spt = np.array([ [0,1,2], [0,1,3], [0,2,3], [1,2,3] ])
  edg = np.array([ [0,1,3], [0,2,4], [1,2,5], [3,4,5] ])
  freq = 1e3
  np.set_printoptions(3)
  for freq in [1e3, 10e3, 100e3, 1e6, 10e6, 100e6, 1e9, 10e9, 100e9]:
    sol = solve(vrt, spt, edg, freq)
    print(sol)
