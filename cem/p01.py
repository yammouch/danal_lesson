import numpy as np

def make_mass(p):
  q = p[:, 1:] - p[:, 0][..., np.newaxis]

  ax = np.array( [ [ 1, 2 ]
                 , [ 2, 0 ]
                 , [ 0, 1 ] ] )
  cb = np.array( [ [ 1, 2, 0 ]
                 , [ 2, 0, 1 ] ] )
  n = np.zeros((3, 4), dtype=np.float64)
  n[:,1:] = q[ax[:,0:1],cb[0]]*q[ax[:,1:2],cb[1]] \
          - q[ax[:,1:2],cb[0]]*q[ax[:,0:1],cb[1]]
  n[:,0] = -n[:,1:].sum(axis=-1)

  return n

if __name__ == '__main__':
  p = np.array \
  ( [ [ 0, 1, 0, 0 ]
    , [ 0, 0, 1, 0 ]
    , [ 0, 0, 0, 1 ] ]
  , dtype=np.float64 )
  print(make_mass(p))
  p = np.array \
  ( [ [ 0, 2, 1, 1 ]
    , [ 1, 1, 2, 1 ]
    , [ 1, 1, 1, 2 ] ]
  , dtype=np.float64 )
  print(make_mass(p))
