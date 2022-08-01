import numpy as np
import p02 as dut

tests = []

def add_test(f):
    tests.append((f.__name__, f))
    return f

@add_test
def test_f_kin_mat():
    basis_nwn = np.arange(-2, 3)[:, None]
    expc \
    = -0.5 \
    * np.array \
      ( [ [  0, 1, 4, 9, 16 ]
        , [  1, 0, 1, 4,  9 ]
        , [  4, 1, 0, 1,  4 ]
        , [  9, 4, 1, 0,  1 ]
        , [ 16, 9, 4, 1,  0 ] ] )
    result = dut.f_kin_mat(basis_nwn, [2*np.pi])
    if (np.abs(result - expc) < 1e-3).all():
        print("[OK] test_f_kin_mat")
    else:
        print('[ER] test_f_kin_mat')
        print(result)

@add_test
def test_symmetrize():
    x = np.array \
    ( [ [  0,  2,  4,  6 ]
      , [  8, 10, 12, 14 ]
      , [ 16, 18, 20, 22 ]
      , [ 24, 26, 28, 30 ] ]
    , dtype=np.float64 )
    expc = np.array \
    ( [ [  0,  2,  2,  6 ]
      , [  8, 10,  6, 14 ]
      , [  8,  9,  5, 11 ]
      , [ 24, 26, 14, 30 ] ] )
    dut.symmetrize(x)
    if (np.abs(x - expc) < 1e-3).all():
        print("[OK] test_symmetrize")
    else:
        print("[ER] test_symmetrize")
        print(x)

@add_test
def test_f_basis_nwn_cart():
    x = np.array \
    ( [ [0, 1]
      , [2, 3] ]
    , dtype=int )
    expc = np.array \
    ( [ [0, 0, 0, 1, 0, 1]
      , [0, 1, 0, 1, 2, 3]
      , [1, 0, 2, 3, 0, 1]
      , [1, 1, 2, 3, 2, 3] ]
    , dtype=int )
    result = dut.f_basis_nwn_cart(x)
    if (result == expc).all():
        print("[OK] test_f_basis_nwn_cart")
    else:
        print("[ER] test_f_basis_nwn_cart")
        print(result)

@add_test
def test_f_vext_nwn():
    x = np.array \
    ( [ [0, 0, -1, 1, -1, 1]
      , [0, 1, -1, 1,  1, 0]
      , [1, 0,  1, 0, -1, 1]
      , [1, 1,  1, 0,  1, 0]
      , [2, 0,  2, 0, -1, 1] ]
    , dtype=int )
    expc = np.array \
    ( [ [0, 0,  0,  0]
      , [0, 1, -2,  1]
      , [1, 0,  2, -1]
      , [1, 1,  0,  0] ]
    , dtype=int )
    result = dut.f_vext_nwn(x, np.zeros((4, 4)))
    if (result == expc).all():
        print("[OK] test_f_vext_nwn")
    else:
        print("[ER] test_f_vext_nwn")
        print(result)

@add_test
def test_f_vext_mat():
    vext_nwn = np.array \
    ( [ [0, 0,  0,  0]
      , [0, 1, -2,  1]
      , [1, 0,  2, -1]
      , [1, 1,  0,  0] ]
    , dtype=int )
    expc = np.array \
    ( [ [ 0, 9]
      , [11, 0] ]
    , dtype=int )
    result = dut.f_vext_mat \
    ( np.zeros((2, 2))
    , np.arange(16).reshape(4, 4)
    , vext_nwn )
    if (result == expc).all():
        print("[OK] test_f_vext_mat")
    else:
        print("[ER] test_f_vext_mat")
        print(result)

for _, f in tests:
    f()
