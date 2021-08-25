import numpy as np

vp = np.array( [ [0,0,0,1,1,2]
               , [1,2,3,2,3,3] ], dtype=np.int64 )

def ntet(p): # p.shape: (3, 4)
  ax = np.array( [ [1,2,0] , [2,0,1] ], dtype=np.int64 )
  cb = np.array( [ [1,2,0] , [2,0,1] ], dtype=np.int64 )
  q = p[:,1:] - p[:,0][...,None] #; print(q)
  n = np.zeros((3,4), dtype=np.float64)
  n[:,1:] = q[ax[0][:,None],cb[0]]*q[ax[1][:,None],cb[1]] \
          - q[ax[1][:,None],cb[0]]*q[ax[0][:,None],cb[1]]
  n[:,0] = -n[:,1:].sum(axis=-1) #; print(n)
  vol = (q[:,0]*n[:,1]).sum(axis=-1)
  n /= vol
  return n, vol

def make_mass(n): # n.shape == (3, 4)
  cc = (  vp[[1,1,0]][...,None  ]
       == vp[[1,0,0]][...,None,:] ) + 1 #; print(cc)
  ma  = (n[:,vp[0]][...,None]*n[:,vp[0]][...,None,:]).sum(axis=-3)*cc[0]
  m1  = (n[:,vp[0]][...,None]*n[:,vp[1]][...,None,:]).sum(axis=-3)*cc[1]
  ma += m1
  ma += np.moveaxis(m1,-2,-1)
  ma += (n[:,vp[1]][...,None]*n[:,vp[1]][...,None,:]).sum(axis=-3)*cc[2]
  return ma

if __name__ == '__main__':
  p = np.array \
  ( [ [ 0, 1, 0, 0 ]
    , [ 0, 0, 1, 0 ]
    , [ 0, 0, 0, 1 ] ]
  , dtype=np.float64 )
  n, vol = ntet(p)
  print(make_mass(n))

  p = np.array \
  ( [ [ 0, 2, 1, 1 ]
    , [ 1, 1, 2, 1 ]
    , [ 1, 1, 1, 2 ] ]
  , dtype=np.float64 )
  n, vol = ntet(p)
  print(make_mass(n))
  print(make_mass(n)*vol*vol)
