import numpy as np
import scipy.sparse
import p01

c  = 299792458.0
u0 = 4e-7*np.pi
e0 = 1/(u0*c**2)

vp = np.array([[0,0,0,1,1,2],[1,2,3,2,3,3]])
vs = np.array([[0,0,1],[1,2,2]])

def local2global(glo, tet, v2e, loc): # inplace
    for i, l in zip(tet, loc):
        dst = v2e[tuple(i[vp])]
        glo[dst, dst.T] += l

def pec(glo, v2e, tri):
    vpairs = np.moveaxis(tri[...,vs],-2,0).reshape(2,-1)
    edge0 = np.array(v2e[vpairs[0], vpairs[1]])[0]
    glo[edge0       ] = 0
    glo[edge0, edge0] = 1

def freq1(freq, tet, tri, stiff, mass, e2v, v2e):
    lhs = np.zeros((e2v.shape[0], e2v.shape[0]), dtype=np.complex128)
    rhs = np.zeros((e2v.shape[0],), dtype=np.complex128)
    local2global(lhs, tet, v2e, stiff/u0)
    local2global(lhs, tet, v2e, -(2*np.pi*freq)**2*e0*mass)
    pec(lhs, v2e, tri)
    rhs[v2e[0,3]] = -2j*np.pi*freq
    sol = np.linalg.solve(lhs, rhs)
    del lhs
    del rhs
    return sol

def main():
    np.set_printoptions(precision=3)
    vrt = np.array \
    ( [ [ 0, 1, 0, 0, 1, 0 ]
      , [ 0, 0, 1, 0, 0, 1 ]
      , [ 0, 0, 0, 1, 1, 1 ] ] )
    tet = np.array \
    ( [ [ 0, 1, 2, 3 ]
      , [ 1, 2, 3, 4 ]
      , [ 2, 3, 4, 5 ] ] )
    tri = np.array \
    ( [ [0, 1, 2]
      , [3, 4, 5] ] )
    e2v = np.unique(tet[:, np.moveaxis(vp,0,1)].reshape(-1,2), axis=0)
    v2e = scipy.sparse.csr_matrix \
    ( ( np.arange(e2v.shape[0])
      , (e2v[:,0], e2v[:,1]) ) )
    coords = np.moveaxis(vrt[:,tet], 0, 1)
    n, vol = p01.ntet(coords)
    stiff = p01.make_stiff(n, vol)
    mass = p01.make_mass(n, vol)
    for freq in [10e3, 100e3, 1e6]:
        sol = freq1(freq, tet, tri, stiff, mass, e2v, v2e)
        print(sol)
        print(1/(e0*0.5*2*np.pi*freq))

if __name__ == '__main__':
    main()