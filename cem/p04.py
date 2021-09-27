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

def isrc(rhs, freq, vrt, nodes, v2e):
    diff = vrt[:,nodes[:,1]] - vrt[:,nodes[:,0]]
    diff = diff.T
    mag = (diff**2).sum(axis=-1)**0.5
    dirc = diff/mag[:,None]
    ip = (dirc*np.array([0,0,1])).sum(axis=-1)
    rhs[v2e[nodes[:,0],nodes[:,1]]] = -2j*np.pi*freq*ip

def isrc_v(sol, vrt, nodes, v2e):
    diff = vrt[:,nodes[:,1]] - vrt[:,nodes[:,0]]
    diff = diff.T
    mag = (diff**2).sum(axis=-1)**0.5
    dirc = diff/mag[:,None]
    ip = (dirc*np.array([0,0,1])).sum(axis=-1)
    return (sol[v2e[nodes[:,0],nodes[:,1]][0]] * ip).sum()

def pec(glo, v2e, tri):
    vpairs = np.moveaxis(tri[...,vs],-2,0).reshape(2,-1)
    edge0 = np.array(v2e[vpairs[0], vpairs[1]])[0]
    glo[edge0       ] = 0
    glo[edge0, edge0] = 1

def air(glo, freq, vrt, tet, v2e):
    coords = np.moveaxis(vrt[:,tet], 0, 1)
    n, vol = p01.ntet(coords)
    stiff = p01.make_stiff(n, vol)
    mass = p01.make_mass(n, vol)
    local2global(glo, tet, v2e, stiff/u0)
    local2global(glo, tet, v2e, -(2*np.pi*freq)**2*e0*mass)

def solve_geom(freq, vrt, pgroups, e2v, v2e):
    lhs = np.zeros((e2v.shape[0], e2v.shape[0]), dtype=np.complex128)
    rhs = np.zeros((e2v.shape[0],), dtype=np.complex128)
    dirichlet = []
    for ptype, attr, nodes in pgroups:
        if ptype == 'v': # volume
            air(lhs, freq, vrt, nodes, v2e)
        elif ptype == 'b': # boundary condition
            dirichlet.append((attr, nodes))
        elif ptype == 'e': # excitation
            isrc(rhs, freq, vrt, nodes, v2e)
        else:
            raise Exception("Unsupported physical type {}".format(ptype))
    for _, nodes in dirichlet:
        pec(lhs, v2e, nodes)
    sol = np.linalg.solve(lhs, rhs)
    del lhs
    del rhs
    print(sol)
    for ptype, attr, nodes in pgroups:
        if ptype == 'e':
            print(isrc_v(sol, vrt, nodes, v2e))
    print(1/(e0*0.5*2*np.pi*freq))

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
    lin = np.array( [ [0, 3] ] )
    e2v = np.unique(tet[:, np.moveaxis(vp,0,1)].reshape(-1,2), axis=0)
    v2e = scipy.sparse.csr_matrix \
    ( ( np.arange(e2v.shape[0])
      , (e2v[:,0], e2v[:,1]) ) )
    pgroups = [('e', (), lin), ('b', (), tri), ('v', (), tet)]
    for freq in [10e3, 100e3, 1e6]:
        solve_geom(freq, vrt, pgroups, e2v, v2e)

if __name__ == '__main__':
    main()