import numpy as np
import p01 as dut

k_basis = np.arange(-2, 3)

def test_kin():
  expc \
  = -0.5 \
  * np.array \
    ( [ [  0, 1, 4, 9, 16 ]
      , [  1, 0, 1, 4,  9 ]
      , [  4, 1, 0, 1,  4 ]
      , [  9, 4, 1, 0,  1 ]
      , [ 16, 9, 4, 1,  0 ] ] )
  result = dut.kin(k_basis[:, None], 2*np.pi)
  if (np.abs(result - expc) < 1e-3).all():
    print("[OK] test_kin")
  else:
    print('[ER] test_kin')
    print(result)

test_kin()
