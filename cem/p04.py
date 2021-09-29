import numpy as np
import scipy.sparse
import scipy.linalg
import p01

c  = 299792458.0
u0 = 4e-7*np.pi
e0 = 1/(u0*c**2)

vp = np.array([[0,0,0,1,1,2],[1,2,3,2,3,3]])
vs = np.array([[0,0,1],[1,2,2]])

def local2global(glo, tet, v2e, loc, bwh): # inplace
    for i, l in zip(tet, loc):
        dst = v2e[tuple(i[vp])]
        if bwh:
            pass
        else:
            glo[dst, dst.T] += l

def local2global2d(glo, tri, v2e, loc, bwh): # inplace
    print(glo.shape)
    for i, l in zip(tri, loc):
        dst = v2e[tuple(i[vs])]
        if bwh:
            pass
        else:
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

def pec(glo, v2e, tri, bwh):
    vpairs = np.moveaxis(tri[...,vs],-2,0).reshape(2,-1)
    edge0 = np.array(v2e[vpairs[0], vpairs[1]])[0]
    if bwh:
        pass
    else:
        glo[edge0       ] = 0
        glo[edge0, edge0] = 1

def air(glo, freq, vrt, tet, v2e, bwh):
    coords = np.moveaxis(vrt[:,tet], 0, 1)
    n, vol = p01.ntet(coords)
    stiff = p01.make_stiff(n, vol)
    mass = p01.make_mass(n, vol)
    local2global(glo, tet, v2e, stiff/u0, bwh)
    local2global(glo, tet, v2e, -(2*np.pi*freq)**2*e0*mass, bwh)

def absorb(lhs, freq, vrt, nodes, v2e, bwh):
    coords = np.moveaxis(vrt[:,nodes], 0, 1)
    n, area = p01.ntri(coords)
    ab = p01.bound(n, area)
    local2global2d(lhs, nodes, v2e, 2j*np.pi*freq*np.sqrt(e0/u0)*ab, bwh)
   #local2global2d(lhs, nodes, v2e, -2j*np.pi*freq*np.sqrt(e0/u0)*ab)

def solve_geom(freq, vrt, pgroups, nedge, v2e, bwh):
    if bwh:
        lhs = np.zeros((2*bwh+1, nedge), dtype=np.complex128)
    else:
        lhs = np.zeros((nedge, nedge), dtype=np.complex128)
    rhs = np.zeros((nedge,), dtype=np.complex128)
    dirichlet = []
    for ptype, attr, nodes in pgroups:
        if ptype == 'v': # volume
            air(lhs, freq, vrt, nodes, v2e, bwh)
        elif ptype == 'b': # boundary condition
            dirichlet.append((attr, nodes))
        elif ptype == 'a': # absorbing boundary
            absorb(lhs, freq, vrt, nodes, v2e, bwh)
        elif ptype == 'e': # excitation
            isrc(rhs, freq, vrt, nodes, v2e)
        else:
            raise Exception("Unsupported physical type {}".format(ptype))
    for _, nodes in dirichlet:
        pec(lhs, v2e, nodes, bwh)
    if bwh:
        sol = scipy.linalg.solve_banded \
        ( (bwh, bwh), lhs, rhs, overwrite_ab=True, overwrite_b=True )
    else:
        sol = np.linalg.solve(lhs, rhs)
    del lhs
    print(sol)
    for ptype, attr, nodes in pgroups:
        if ptype == 'e':
            print(isrc_v(sol, vrt, nodes, v2e))
    print(1/(e0*0.5*2*np.pi*freq))

def edge_num(tet):
    e2v = np.unique(tet[:, np.moveaxis(vp,0,1)].reshape(-1,2), axis=0)
    v2e = scipy.sparse.csr_matrix \
    ( ( np.arange(e2v.shape[0])
      , (e2v[:,0], e2v[:,1]) ) )
    lil = scipy.sparse.lil_matrix \
    ( (e2v.shape[0], e2v.shape[0]), dtype=np.int64 )
    print(tet[:,vp[0]])
    print(tet[:,vp[1]])
    tet_e = v2e[tet[:,vp[0]], tet[:,vp[1]]].toarray()
    print(tet_e)
    for v in tet_e:
        lil[v[:, None], v] = 1
    print(lil.toarray())
    perm = scipy.sparse.csgraph.reverse_cuthill_mckee(lil.tocsr())
    print(perm)
    print(lil[perm[:,None], perm].toarray())

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
        solve_geom(freq, vrt, pgroups, e2v.shape[0], v2e, None)

if __name__ == '__main__':
    main()