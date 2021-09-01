import numpy as np
import p01 as dut

if __name__ == '__main__':
  np.set_printoptions(3)
  p = np.array \
  ( [ [ [  1,  2,  1,  1 ]
      , [  0,  0,  1,  0 ]
      , [ -1, -1, -1,  0 ] ]
  # , [ [ 0, 2*3**0.5, 0, 3**0.5   ]
  #   , [ 0, 0       , 3, 1        ]
  #   , [ 0, 0       , 0, 2*2**0.5 ] ] ]
    , [ [  0, 2**0.5,  2**0.5, 0 ]
      , [  0, 1     , -1     , 0 ]
      , [ -1, 0     ,  0     , 1 ] ] ]
  , dtype=np.float64 )
  n, vol = dut.ntet(p)
  m = dut.make_mass(n, vol)
  print(vol)
  print(n)
  print(m / vol[..., None, None] * 120)
  s = dut.make_stiff(n, vol)
  print(s / vol[..., None, None])
  n, area = dut.ntri(p[..., 0:3])
  print(n, area)
  print(dut.bound(n, area))
  print(dut.make_b(p[1]))
