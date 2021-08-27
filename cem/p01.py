import numpy as np

vp = np.array( [ [0,0,0,1,1,2]
               , [1,2,3,2,3,3] ], dtype=np.int64 )

def ntet(p): # p.shape: (..., 3, 4)
  q = p[..., 1:] - p[..., 0][..., None] #; print(q)
  n = np.empty_like(p, dtype=np.float64)
  n[..., 1:] \
  = q[..., [[1],[2],[0]], [1,2,0]] * q[..., [[2],[0],[1]], [2,0,1]] \
  - q[..., [[2],[0],[1]], [1,2,0]] * q[..., [[1],[2],[0]], [2,0,1]]
  n[..., 0] = -n[..., 1:].sum(axis=-1) #; print(n)
  vol = (q[..., 0]*n[..., 1]).sum(axis=-1)
  n /= vol[..., None, None]
  return n, vol

def make_stiff(n, vol): # n.shape: (..., 3, 4)
  np = n[..., [[1],[2],[0]], vp[0]] * n[..., [[2],[0],[1]], vp[1]] \
     - n[..., [[2],[0],[1]], vp[0]] * n[..., [[1],[2],[0]], vp[1]]
  ip = ( np[..., None] * np[..., None, :] ).sum(axis=-3)
  ip *= vol[..., None, None]
  return ip

def make_mass(n, vol): # n.shape: (..., 3, 4)
  cc = (  vp[[1,1,0]][..., None   ]
       == vp[[1,0,0]][..., None, :] ) + 1 #; print(cc)
  ma  = ( n[..., vp[0]][..., None   ]
        * n[..., vp[0]][..., None, :] ).sum(axis=-3) * cc[0]
  m1  = ( n[..., vp[0]][..., None   ]
        * n[..., vp[1]][..., None, :] ).sum(axis=-3) * cc[1]
  ma += m1
  ma += np.moveaxis(m1,-2,-1)
  ma += ( n[..., vp[1]][..., None   ]
        * n[..., vp[1]][..., None, :] ).sum(axis=-3) * cc[2]
  ma *= vol[..., None, None]/120
  return ma

def ntri(p): # p: (..., 3, 3)
  q = p[..., 1:] - p[..., [0]] # (..., 3, 2)
  n = np.empty_like(p, dtype=np.float64)
  ip = q.prod(axis=-1).sum(axis=-1)
  sf = ip[..., None]/((q**2).sum(axis=-2))
  n[..., 1:] = q
  n[..., 1:] -= sf[..., None, :] * q[..., ::-1]
  n[..., 0] = -n[..., 1:].sum(axis=-1)
  n /= (n**2).sum(axis=-2)[..., None, :]
  op = q[..., [1,2,0], 0]*q[..., [2,0,1], 1] \
     - q[..., [2,0,1], 0]*q[..., [1,2,0], 1]
  area = np.sqrt((op**2).sum(axis=-1))
  return n, area

def bound(n, area): # n: (..., 3, 3), area: (...)
  a0 = (n[..., [[0],[0],[1]]] * n[..., [[0,0,1]]]).sum(axis=-3)
  a0[..., [0, 1, 1, 2, 2], [0, 1, 2, 1, 2]] *= 2
  a1 = (n[..., [[0],[0],[1]]] * n[..., [[1,2,2]]]).sum(axis=-3)
  a1[..., [2, 2], [0, 1]] *= 2
  a2 = (n[..., [[1],[2],[2]]] * n[..., [[0,0,1]]]).sum(axis=-3)
  a2[..., [0, 1], [2, 2]] *= 2
  a3 = (n[..., [[1],[2],[2]]] * n[..., [[1,2,2]]]).sum(axis=-3)
  a3[..., [0, 0, 1, 1, 2], [0, 1, 0, 1, 2]] *= 2
  return (a0 + a1 + a2 + a3)*(area[..., None, None]/24)


if __name__ == '__main__':
  u0 = 4e-7*np.pi
  e0 = 8.854e-12
  vrt = np.array \
  ( [ [ 0, 2*3**0.5, 0, 3**0.5 ]
    , [ 0, 0       , 3, 1      ]
    , [ 0, 0       , 0, 2**0.5 ] ] )
  spt = np.array([ [0,1,2], [0,1,3], [0,2,3], [1,2,3] ])
  edg = np.array([ [0,1,3], [0,2,4], [1,2,5], [3,4,5] ])
  b = np.zeros((6, 6), dtype=np.complex128)
  freq = 1e3
  for p, e in zip(spt, edg):
    n, area = ntri(vrt[:, p])
    b1 = bound(n, area)
    b[e[:,None],e[None,:]] += 2j*np.pi*freq*np.sqrt(1/(u0*e0))*b1
  n, vol = ntet(vrt)
  stiff = make_stiff(n, vol)
  mass = make_mass(n, vol)
  lhs = stiff/(4e-7*np.pi) - (2*np.pi*freq)**2*e0*mass + b
  sol = np.linalg.solve(lhs, np.zeros(6, dtype=np.complex128))
  print(sol)
