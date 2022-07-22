import numpy as np
import itertools
import p01 as dut

tests = []

def add_test(f):
    tests.append((f.__name__, f))

@add_test
def test_kin():
    k_basis = np.arange(-2, 3)[:, None]
    expc \
    = -0.5 \
    * np.array \
      ( [ [  0, 1, 4, 9, 16 ]
        , [  1, 0, 1, 4,  9 ]
        , [  4, 1, 0, 1,  4 ]
        , [  9, 4, 1, 0,  1 ]
        , [ 16, 9, 4, 1,  0 ] ] )
    result = dut.kin(k_basis, [2*np.pi])
    if (np.abs(result - expc) < 1e-3).all():
        print("[OK] test_kin")
    else:
        print('[ER] test_kin')
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
def test_cou_idx():
    x = np.array([-1, 0, 1])[:, None]
    expc = \
    [ [ [ np.array([0, 1, 2])
        , np.array([0, 1, 2])
        , np.array([1, 1, 1]) ]
      , [ np.array([1, 2])
        , np.array([0, 1])
        , np.array([2, 2]) ]
      , [ np.array([], dtype=np.int64)
        , np.array([], dtype=np.int64)
        , np.array([], dtype=np.int64) ] ]
    , [ [ np.array([0, 1])
        , np.array([1, 2])
        , np.array([0, 0]) ]
      , [ np.array([0, 1, 2])
        , np.array([0, 1, 2])
        , np.array([1, 1, 1]) ]
      , [ np.array([1, 2])
        , np.array([0, 1])
        , np.array([2, 2]) ] ]
    , [ [ np.array([], dtype=np.int64)
        , np.array([], dtype=np.int64)
        , np.array([], dtype=np.int64) ]
      , [ np.array([0, 1])
        , np.array([1, 2])
        , np.array([0, 0]) ]
      , [ np.array([0, 1, 2])
        , np.array([0, 1, 2])
        , np.array([1, 1, 1]) ] ] ]
    result = dut.cou_idx(x)
    ok = True
    for i, j, k in itertools.product(range(3), range(3), range(3)):
        if (expc[i][j][k] == result[i][j][k]).all():
            pass
        else:
            ok = False
            break
    if ok:
        print("[OK] test_cou_idx")
    else:
        print("[ER] test_cou_idx")
        print(result)


for _, f in tests:
    f()
