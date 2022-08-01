import numpy as np

def kin(basis_nwn, box_size):
    k = 2*np.pi/np.array(box_size)[None, :]*basis_nwn
    x = k[:, None, :] - k[None, :, :]
    x = x**2
    x = x.sum(axis=-1)
    x = -0.5*x
    return x

def symmetrize(a):
    for i in range(a.ndim):
        l = a.shape[i]
        if l % 2 == 0:
            a[(slice(None),)*i + (l//2,)] *= 0.5
    return a

def f_basis_nwn_cart(basis_nwn):
    rv = np.empty((basis_nwn.shape[0]**2, 2+2*basis_nwn.shape[1]), dtype=int)
    rv[:, :2] = \
    np.mgrid[:basis_nwn.shape[0], :basis_nwn.shape[0]] \
    .reshape(2, -1).T
    rv[:, 2                   :2+  basis_nwn.shape[1]] = basis_nwn[rv[:,0]]
    rv[:, 2+basis_nwn.shape[1]:2+2*basis_nwn.shape[1]] = basis_nwn[rv[:,1]]
    return rv

def f_vext_nwn(basis_nwn_cart, vext):
    l   =  basis_nwn_cart.shape[0]
    dim = (basis_nwn_cart.shape[1] - 2)//2
    x = np.empty((l, 2+dim), dtype=int)
    x[:,:2 ] = basis_nwn_cart[:, :2    ]
    x[:, 2:] = basis_nwn_cart[:,2:2+dim] - basis_nwn_cart[:,2+dim:]
    rv = x[(np.abs(x[:, 2:]) <= np.array(vext.shape)//2).all(axis=1)]
    return rv
