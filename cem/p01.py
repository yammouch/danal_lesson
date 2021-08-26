import numpy as np

vp = np.array( [ [0,0,0,1,1,2]
               , [1,2,3,2,3,3] ], dtype=np.int64 )

def ntet(p): # p.shape: (..., 3, 4)
  q = p[..., 1:] - p[..., 0][..., None] #; print(q)
  n = np.zeros_like(p, dtype=np.float64)
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

if __name__ == '__main__':
  p = np.array \
  ( [ [ [  1,  2,  1,  1 ]
      , [  0,  0,  1,  0 ]
      , [ -1, -1, -1,  0 ] ]
    , [ [ 0, 2*3**0.5, 0, 3**0.5 ]
      , [ 0, 0       , 3, 1      ]
      , [ 0, 0       , 0, 2**0.5 ] ] ]
  , dtype=np.float64 )
  n, vol = ntet(p)
  m = make_mass(n, vol)
  print(vol)
  print(m / vol[..., None, None] * 120)
  s = make_stiff(n, vol)
  print(s / vol[..., None, None])

 #p = np.array \
 #( [ [ 0, 2, 1, 1 ]
 #  , [ 1, 1, 2, 1 ]
 #  , [ 1, 1, 1, 2 ] ]
 #, dtype=np.float64 )
 #n, vol = ntet(p)
 #print(make_mass(n))
 #print(make_mass(n)*vol*vol)
