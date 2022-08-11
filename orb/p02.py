import numpy as np
import scipy.linalg
import datetime

def f_kin_mat(basis_nwn, box_size):
    box_size = np.array(box_size)
    k = 2*np.pi/box_size*basis_nwn
    rv = (k**2).sum(axis=-1)
    rv *= 0.5
    return np.diag(rv)

def symmetrize(a):
    for i in range(a.ndim):
        l = a.shape[i]
        if l % 2 == 0:
            a[(slice(None),)*i + (l//2,)] *= 0.5
    return a

def f_basis_nwn_cart(basis_nwn):
    l   = basis_nwn.shape[0]
    dim = basis_nwn.shape[1]
    rv = np.empty((l**2, 2+2*dim), dtype=int)
    rv[:, :2] = np.mgrid[:l, :l].reshape(2, -1).T
    rv[:, 2    :2+dim] = basis_nwn[rv[:,0]]
    rv[:, 2+dim:     ] = basis_nwn[rv[:,1]]
    return rv

def f_vext_nwn(basis_nwn_cart, vext_rc):
    l   =  basis_nwn_cart.shape[0]
    dim = (basis_nwn_cart.shape[1] - 2)//2
    x = np.empty((l, 2+dim), dtype=int)
    x[:,:2 ] = basis_nwn_cart[:, :2    ]
    x[:, 2:] = basis_nwn_cart[:,2:2+dim] - basis_nwn_cart[:,2+dim:]
    rv = x[(np.abs(x[:, 2:]) <= np.array(vext_rc.shape)//2).all(axis=1)]
    return rv

def f_vext_mat(basis_nwn, vext_rc, vext_nwn):
    l = basis_nwn.shape[0]
    rv = np.zeros((l, l), dtype=np.complex128)
    rv[vext_nwn[:,0], vext_nwn[:,1]] = vext_rc[tuple(vext_nwn[:,2:].T)]
    return rv

def solve_1elec(basis_nwn, box_size, vext):
    t0 = datetime.datetime.now()
    vext_rc = np.fft.fftn(vext, norm="forward")
    t1 = datetime.datetime.now()
    print('np.fft.fftn', t1 - t0)
    t0 = t1
    symmetrize(vext_rc)
    t1 = datetime.datetime.now()
    print('symmetrize', t1 - t0)
    t0 = t1
    basis_nwn_cart = f_basis_nwn_cart(basis_nwn)
    t1 = datetime.datetime.now()
    print('f_basis_nwn_cart', t1 - t0)
    t0 = t1
    vext_nwn = f_vext_nwn(basis_nwn_cart, vext_rc)
    t1 = datetime.datetime.now()
    print('f_vext_nwn', t1 - t0)
    t0 = t1
    vext_mat = f_vext_mat(basis_nwn, vext_rc, vext_nwn)
    t1 = datetime.datetime.now()
    print('f_vext_mat', t1 - t0)
    t0 = t1
    kin_mat = f_kin_mat(basis_nwn, box_size)
    t1 = datetime.datetime.now()
    print('f_kin_mat', t1 - t0)
    t0 = t1
    lhs = kin_mat + vext_mat
    e, v = scipy.linalg.eigh \
    ( lhs
    , subset_by_index=[0, min(6, len(lhs))-1] )
    t1 = datetime.datetime.now()
    print('eigh', t1 - t0)
    t0 = t1
    return e, v

def symmetrize4move(a):
    for i, n in enumerate(a.shape):
        if n % 2 != 0:
            continue
        a = np.concatenate \
        ( [     a[(slice(None),)*i + (slice(0,n//2)         ,)]
          , 0.5*a[(slice(None),)*i + ([n//2, n//2]          ,)]
          ,     a[(slice(None),)*i + (slice(-(n//2)+1, None),)] ]
        , axis=i )
    return a

def dist2(box_size, vext):
    rv = np.array(0, dtype=float)
    for l, n in zip(box_size, vext.shape):
        a = np.arange(-(n//2), -(n//2)+n)+0.5
        a *= l/n
        #a = np.linspace(l/n*(-(n//2)+0.5), l/n*(-(n//2)-0.5)+l, n)
        rv = rv[..., None] + a**2
    return rv
