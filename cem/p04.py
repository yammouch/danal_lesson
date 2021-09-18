import numpy as np
import scipy.sparse
import p01

c  = 299792458.0
u0 = 4e-7*np.pi
e0 = 1/(u0*c**2)

def freq1(freq, tet, stiff, mass, e2v, v2e):
    lhs = np.zeros((e2v.shape[0], e2v.shape[0]), dtype=np.complex128)
    rhs = np.zeros((e2v.shape[0],), dtype=np.complex128)
    for i, s, m in zip(tet, stiff, mass):
        dst = v2e[tuple(i[vp])]
        lhs[dst, dst.T] += s/u0
        lhs[dst, dst.T] -= (2*np.pi*freq)**2*e0*m
    edge0 = np.array(v2e[ [0,0,1,    3,3,4]
                        , [1,2,2,    4,5,5] ])[0]
   #edge0 = np.array(v2e[ [0,0,1,1,2,3,3,4]
   #                    , [1,2,2,4,5,4,5,5] ])[0]
    lhs[edge0       ] = 0
    lhs[edge0, edge0] = 1
   #print(lhs)
    rhs[v2e[0,3]] = -2j*np.pi*freq
   #print(rhs)
    sol = np.linalg.solve(lhs, rhs)
    return sol

vp = np.array([[0,0,0,1,1,2],[1,2,3,2,3,3]])

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
    e2v = np.unique(tet[:, np.moveaxis(vp,0,1)].reshape(-1,2), axis=0)
    v2e = scipy.sparse.csr_matrix \
    ( ( np.arange(e2v.shape[0])
      , (e2v[:,0], e2v[:,1]) ) )
   #print(v2e)
    coords = np.moveaxis(vrt[:,tet], 0, 1)
    n, vol = p01.ntet(coords)
    stiff = p01.make_stiff(n, vol)
    mass = p01.make_mass(n, vol)
    for freq in [10e3, 100e3, 1e6]:
        sol = freq1(freq, tet, stiff, mass, e2v, v2e)
        print(sol)
        print(1/(e0*0.5*2*np.pi*freq))

if __name__ == '__main__':
    main()