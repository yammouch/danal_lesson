import numpy as np
import p01 as dut

np.set_printoptions(precision=3)

p = np.array \
( [ [ [0, 1, 0, 0]
    , [0, 0, 1, 0]
    , [0, 0, 0, 1] ]
  , [ [ 0, 2**(1/2),  2**(1/2), 0]
    , [ 0,        1,        -1, 0]
    , [-1,        0,         0, 1] ]
  , [ [ 0, 7,  7, 0]
    , [ 0, 7, -7, 0]
    , [-1, 0,  0, 1] ] ] )

if True:
    exp_n = np.array \
    ( [ [ [-1, 1, 0, 0]
        , [-1, 0, 1, 0]
        , [-1, 0, 0, 1] ]
      , [ [ -2**(1/2)/4, 2**(1/2)/4, 2**(1/2)/4, -2**(1/2)/4]
        , [         0  ,        1/2,       -1/2,         0  ]
        , [        -1/2,        0  ,        0  ,         1/2] ]
      , [ [ -1/14, 1/14,  1/14, -1/14]
        , [  0   , 1/14, -1/14,  0   ]
        , [ -1/2 , 0   ,  0   ,  1/2 ] ] ] )
    exp_vol = np.array \
    ( [ 1, -4*2**(1/2), -196 ] )
    n, vol = dut.ntet(p)
    print('n of ntet ', end='')
    if (np.abs(n-exp_n) < 1e-3).all():
        print("[OK]")
    else:
        print("[ER]")
        print(n)
    print('vol of ntet ', end='')
    if (np.abs(vol-exp_vol) < 1e-3).all():
        print("[OK]")
    else:
        print("[ER]")
        print(vol)

if True:
    n, vol = dut.ntet(p)
    exp = np.array \
    ( [ [ [1/12, 1/24, 1/24,  0    , 0    ,  0    ]
        , [1/24, 1/12, 1/24,  0    , 0    ,  0    ]
        , [1/24, 1/24, 1/12,  0    , 0    ,  0    ]
        , [0   , 0   , 0   ,  1/30 , 1/120, -1/120]
        , [0   , 0   , 0   ,  1/120, 1/30 ,  1/120]
        , [0   , 0   , 0   , -1/120, 1/120,  1/30 ] ] ] )
    n, vol = dut.ntet(p)
    m = dut.make_mass(n, vol)
    print('make_mass ', end='')
    if (np.abs(m[0]-exp[0]) < 1e-3).all():
        print("[OK]")
    else:
        print("[ER]")
        print(m)


#if __name__ == '__main__':
if False:
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
  print(m)
  s = dut.make_stiff(n, vol)
  print(s / vol[..., None, None])
  n, area = dut.ntri(p[..., 0:3])
  print(n, area)
  print(dut.bound(n, area))
  print(dut.make_b(p[1]))
