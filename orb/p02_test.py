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
    = 0.5 \
    * np.array \
      ( [ [ 4, 0, 0, 0, 0 ]
        , [ 0, 1, 0, 0, 0 ]
        , [ 0, 0, 0, 0, 0 ]
        , [ 0, 0, 0, 1, 0 ]
        , [ 0, 0, 0, 0, 4 ] ] )
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

@add_test
def test_symmetrize4move():
    x = np.array \
    ( [ [  0,  2,  4,  6 ]
      , [  8, 10, 12, 14 ]
      , [ 16, 18, 20, 22 ]
      , [ 24, 26, 28, 30 ] ]
    , dtype=np.float64 )
    expc = np.array \
    ( [ [  0,  2,  2,  2,  6 ]
      , [  8, 10,  6,  6, 14 ]
      , [  8,  9,  5,  5, 11 ]
      , [  8,  9,  5,  5, 11 ]
      , [ 24, 26, 14, 14, 30 ] ] )
    result = dut.symmetrize4move(x)
    if (np.abs(result - expc) < 1e-3).all():
        print("[OK] test_symmetrize4move")
    else:
        print("[ER] test_symmetrize4move")
        print(x)

@add_test
def test_dist2():
    box_size = [6, 8]
    vext = np.zeros((3, 4))
    expc = np.array \
    ( [ [ 10,  2,  2, 10 ]
      , [ 10,  2,  2, 10 ]
      , [ 18, 10, 10, 18 ] ] )
    result = dut.dist2(box_size, vext)
    if (np.abs(result - expc) < 1e-3).all():
        print("[OK] test_dist2")
    else:
        print("[ER] test_dist2")
        print(result)

@add_test        
def test_oversample_rc():
    x = np.array \
    ( [ [  0,  2,  2,  2,  6 ]
      , [  8, 10,  6,  6, 14 ]
      , [  8,  9,  5,  5, 11 ]
      , [  8,  9,  5,  5, 11 ]
      , [ 24, 26, 14, 14, 30 ] ] )
    expc = np.array \
    ( [ [  0,  2,  2,  0,  0,  0,  2,  6 ]
      , [  8, 10,  6,  0,  0,  0,  6, 14 ]
      , [  8,  9,  5,  0,  0,  0,  5, 11 ]
      , [  0,  0,  0,  0,  0,  0,  0,  0 ]
      , [  0,  0,  0,  0,  0,  0,  0,  0 ]
      , [  0,  0,  0,  0,  0,  0,  0,  0 ]
      , [  8,  9,  5,  0,  0,  0,  5, 11 ]
      , [ 24, 26, 14,  0,  0,  0, 14, 30 ] ] )
    result = dut.oversample_rc(x, [0, 1])
    if (np.abs(result - expc) < 1e-3).all():
        print("[OK] test_oversample_rc")
    else:
        print("[ER] test_oversample_rc")
        print(result)

@add_test
def test_slide_half_grid():
    x = np.arange(16).reshape(4, 4)
    X = np.fft.fftn(x, norm='forward')
    X2 = dut.symmetrize4move(X)
    X3 = dut.slide_half_grid(X2)
    X4 = dut.oversample_rc(X3, [0, 1])
    xo = np.fft.ifftn(X4, norm='forward')
    result = xo[1::2, 1::2]
    if (np.abs(result - x) < 1e-3).all():
        print("[OK] test_slide_half_grid")
    else:
        print("[ER] test_slide_half_grid")
        print(result)

def main():
    for _, f in tests:
        f()

if __name__ == '__main__':
    main()
