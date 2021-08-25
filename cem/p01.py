import numpy as np

def make_mass(p):
  q = p[:, 1:] - p[:, 0][..., np.newaxis]
  return q

if __name__ == '__main__':
  p = np.array \
  ( [ [ 0, 1, 0, 0 ]
    , [ 0, 0, 1, 0 ]
    , [ 0, 0, 0, 1 ] ]
  , dtype=np.float64 )
  print(make_mass(p))
  p = np.array \
  ( [ [ 1, 2, 1, 1 ]
    , [ 1, 1, 3, 1 ]
    , [ 2, 1, 1, 4 ] ]
  , dtype=np.float64 )
  print(make_mass(p))
