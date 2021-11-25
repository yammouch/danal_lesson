import numpy as np
import p01 as dut

#np.set_printoptions(precision=3)

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
    ( [ [ [-1, -1, -1]
        , [ 1,  0,  0]
        , [ 0,  1,  0]
        , [ 0,  0,  1] ]
      , [ [-2**(1/2)/4,  0  , -1/2]
        , [ 2**(1/2)/4,  1/2,  0  ]
        , [ 2**(1/2)/4, -1/2,  0  ]
        , [-2**(1/2)/4,  0  ,  1/2] ]
      , [ [-1/14,  0   , -1/2]
        , [ 1/14,  1/14,  0  ]
        , [ 1/14, -1/14,  0  ]
        , [-1/14,  0   ,  1/2] ] ] )
    exp_vol = np.array \
    ( [ 1, -4*2**(1/2), -196 ] )
    n, vol = dut.ntet(np.moveaxis(p, -2, -1))
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
    exp = np.array \
    ( [ [ [ 10,  5,  5,  0,  0,  0]
        , [  5, 10,  5,  0,  0,  0]
        , [  5,  5, 10,  0,  0,  0]
        , [  0,  0,  0,  4,  1, -1]
        , [  0,  0,  0,  1,  4,  1]
        , [  0,  0,  0, -1,  1,  4] ]
      , [ [ 14,  3,  3, -3, -3,  0]
        , [  3, 14,  3,  3,  0, -3]
        , [  3,  3, 14,  0,  3,  3]
        , [ -3,  3,  0, 14,  3, -3]
        , [ -3,  0,  3,  3, 14,  3]
        , [  0, -3,  3, -3,  3, 14] ]
      , [ [106, 52, 97, -1, 92, 46]
        , [ 52,106, 97,  1, 46, 92]
        , [ 97, 97,296,  0, 97, 97]
        , [ -1,  1,  0,  8,  1, -1]
        , [ 92, 46, 97,  1,106, 52]
        , [ 46, 92, 97, -1, 52,106] ] ]
    , dtype=np.float64 )
    exp[0] /= 120
    exp[1] /= 120*2**0.5
    exp[2] /= 120
    n, vol = dut.ntet(np.moveaxis(p, -2, -1))
    m = dut.make_mass(np.moveaxis(n, -2, -1), vol)
    print('make_mass ', end='')
    if (np.abs(m-exp) < 1e-3).all():
        print("[OK]")
    else:
        print("[ER]")
        print(m)
        print(exp)
        print(np.abs(m-exp))

if True:
    exp = np.array \
    ( [ [ [  2, -1, -1,  1,  1,  0]
        , [ -1,  2, -1, -1,  0,  1]
        , [ -1, -1,  2,  0, -1, -1]
        , [  1, -1,  0,  1,  0,  0]
        , [  1,  0, -1,  0,  1,  0]
        , [  0,  1, -1,  0,  0,  1] ]
      , [ [  2, -1, -1,  1,  1,  0]
        , [ -1,  2, -1, -1,  0,  1]
        , [ -1, -1,  2,  0, -1, -1]
        , [  1, -1,  0,  2, -1,  1]
        , [  1,  0, -1, -1,  2, -1]
        , [  0,  1, -1,  1, -1,  2] ]
      , [ [ 99, -1,-98,  2, 97,  1]
        , [ -1, 99,-98, -2,  1, 97]
        , [-98,-98,196,  0,-98,-98]
        , [  2, -2,  0,  4, -2,  2]
        , [ 97,  1,-98, -2, 99, -1]
        , [  1, 97,-98,  2, -1, 99] ] ]
    , dtype=np.float64 )
    exp[0] *= 4/6
    exp[1] *= 2**0.5/6
    exp[2] /= 49*6
    n, vol = dut.ntet(np.moveaxis(p, -2, -1))
    m = dut.make_stiff(np.moveaxis(n, -2, -1), vol)
   #print(m)
    print('make_stiff ', end='')
    if (np.abs(m-exp) < 1e-3).all():
        print("[OK]")
    else:
        print("[ER]")
        print(m)
        print(exp)
        print(np.abs(m-exp))

if True:
    n, jacob = dut.ntri2(np.moveaxis(2*p[0, :, [[0, 1, 2], [0, 1, 3]]], -1, -2))
    exp_n = np.array \
    ( [ [ [ -0.5,  0.5, 0   ]
        , [ -0.5,  0  , 0.5 ]
        , [ -0  ,  0  , 0   ] ]
      , [ [ -0.5,  0.5, 0   ]
        , [  0  ,  0  , 0   ]
        , [ -0.5,  0  , 0.5 ] ] ] )
    exp_jacob = np.array([4, 4])
    print('n of ntri2 ', end='')
    if (np.abs(n-exp_n) < 1e-3).all():
        print("[OK]")
    else:
        print("[ER]")
        print(n)
    print('jacob of ntri2 ', end='')
    if (np.abs(jacob-exp_jacob) < 1e-3).all():
        print("[OK]")
    else:
        print("[ER]")
        print(jacob)

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
  n, vol = dut.ntet(np.moveaxis(p, -2, -1))
  m = dut.make_mass(np.moveaxis(n, -2, -1), vol)
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
