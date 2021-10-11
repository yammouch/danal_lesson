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
            glo[dst.T + bwh - dst, dst] += l
        else:
            glo[dst.T, dst] += l

def local2global2d(glo, tri, v2e, loc, bwh): # inplace
    print(glo.shape)
    for i, l in zip(tri, loc):
        dst = v2e[tuple(i[vs])]
        if bwh:
            glo[dst.T + bwh - dst, dst] += l
        else:
            glo[dst.T, dst] += l

def isrc(rhs, freq, vrt, nodes, v2e, isrc_dir):
    diff = vrt[:,nodes[:,1]] - vrt[:,nodes[:,0]]
    diff = diff.T
    mag = (diff**2).sum(axis=-1)**0.5
    dirc = diff/mag[:,None]
    ip = (dirc*isrc_dir).sum(axis=-1)
    rhs[v2e[nodes[:,0],nodes[:,1]]] = -2j*np.pi*freq*ip

def isrc_v(sol, vrt, nodes, v2e, isrc_dir):
    diff = vrt[:,nodes[:,1]] - vrt[:,nodes[:,0]]
    diff = diff.T
    mag = (diff**2).sum(axis=-1)**0.5
    dirc = diff/mag[:,None]
    ip = (dirc*isrc_dir).sum(axis=-1)
    return (sol[v2e[nodes[:,0],nodes[:,1]][0]] * ip).sum()

def pec(glo, rhs, v2e, tri, bwh):
    vpairs = np.moveaxis(tri[...,vs],-2,0).reshape(2,-1)
    edge0 = np.array(v2e[vpairs[0], vpairs[1]])[0]
    if bwh:
        for e in edge0:
            glo[ range(glo.shape[0])
               , np.arange(e+bwh, e-bwh-1, -1)%(glo.shape[1]) ] = 0
            glo[bwh, e] = 1
    else:
        glo[edge0       ] = 0
        glo[edge0, edge0] = 1
    rhs[edge0] = 0

def air(glo, freq, vrt, tet, v2e, bwh):
    coords = np.moveaxis(vrt[:,tet], 0, 1)
    n, vol = p01.ntet(coords)
    stiff = p01.make_stiff(n, vol)
    mass = p01.make_mass(n, vol)
    local2global(glo, tet, v2e, stiff/u0, bwh)
    local2global(glo, tet, v2e, -(2*np.pi*freq)**2*e0*mass, bwh)

def cond(glo, freq, vrt, tet, v2e, bwh):
    coords = np.moveaxis(vrt[:,tet], 0, 1)
    n, vol = p01.ntet(coords)
    stiff = p01.make_stiff(n, vol)
    mass = p01.make_mass(n, vol)
    local2global(glo, tet, v2e, stiff/u0, bwh)
    w = 2*np.pi*freq
    local2global(glo, tet, v2e, -w*(w*e0-1j/140e-8)*mass, bwh)

def absorb(lhs, freq, vrt, nodes, v2e, bwh):
    coords = np.moveaxis(vrt[:,nodes], 0, 1)
    n, area = p01.ntri(coords)
    ab = p01.bound(n, area)
    local2global2d(lhs, nodes, v2e, 2j*np.pi*freq*np.sqrt(e0/u0)*ab, bwh)
   #local2global2d(lhs, nodes, v2e, -2j*np.pi*freq*np.sqrt(e0/u0)*ab)

def isrc_2d(rhs, freq, vrt, nodes, v2e, i_density):
    coords = np.moveaxis(vrt[:,nodes], 0, 1)
    n, jacob = p01.ntri2(coords)
    x  = n[..., vs[1]] - n[..., vs[0]]
    x *= np.array(i_density)[..., None]
    x  = x.sum(axis=-2)
    x *= jacob[..., None]
    x /= 6
    x = -2j*np.pi*freq*x
    for i, y in zip(nodes, x):
        dst = v2e[tuple(i[vs])]
        rhs[dst[0]] += y

def isrc_3d(rhs, freq, vrt, nodes, v2e, i_density):
    coords = np.moveaxis(vrt[:,nodes], 0, 1)
   #print(coords)
    n, jacob = p01.ntet(coords)
   #print(n)
   #print(jacob)
    x  = n[..., vp[1]] - n[..., vp[0]]
   #print(x)
    x *= np.array(i_density)[..., None]
   #print(x)
    x  = x.sum(axis=-2)
   #print(x)
    x *= np.abs(jacob[..., None])
   #print(x)
    x /= 24
   #print(x)
    x = -2j*np.pi*freq*x
   #print(x)
    for i, y in zip(nodes, x):
        dst = v2e[tuple(i[vp])]
        rhs[dst[0]] += y
   #print(rhs)

def solve_geom(freq, vrt, pgroups, nedge, v2e, bwh):
    if bwh:
        lhs = np.zeros((2*bwh+1, nedge), dtype=np.complex128)
    else:
        lhs = np.zeros((nedge, nedge), dtype=np.complex128)
    rhs = np.zeros((nedge,), dtype=np.complex128)
    dirichlet = []
    for ptype, attr, nodes in pgroups:
        if ptype == 'v': # air
            air(lhs, freq, vrt, nodes, v2e, bwh)
        elif ptype == 'c': # conductor
            cond(lhs, freq, vrt, nodes, v2e, bwh)
        elif ptype == 'b': # boundary condition
            dirichlet.append((attr, nodes))
        elif ptype == 'a': # absorbing boundary
            absorb(lhs, freq, vrt, nodes, v2e, bwh)
        elif ptype == 'e': # excitation
            isrc(rhs, freq, vrt, nodes, v2e, attr[0])
        elif ptype == 'e2': # excitation 2D
            isrc_2d(rhs, freq, vrt, nodes, v2e, attr[0])
        elif ptype == 'e3': # excitation 3D
            isrc_3d(rhs, freq, vrt, nodes, v2e, attr[0])
        elif ptype == 'p': # probe
            pass
        else:
            raise Exception("Unsupported physical type {}".format(ptype))
    for _, nodes in dirichlet:
        pec(lhs, rhs, v2e, nodes, bwh)
   #print(lhs)
   #print(rhs)
    if bwh:
        sol = scipy.linalg.solve_banded \
        ( (bwh, bwh), lhs, rhs, overwrite_ab=True, overwrite_b=True )
    else:
        sol = np.linalg.solve(lhs, rhs)
    del lhs
    return sol

def edge_num_naive(tet):
    e2v = np.unique(tet[:, np.moveaxis(vp,0,1)].reshape(-1,2), axis=0)
    v2e = scipy.sparse.csr_matrix \
    ( ( np.arange(e2v.shape[0])
      , (e2v[:,0], e2v[:,1]) ) )
    return v2e, e2v

def edge_num_banded(tet):
    v2e, e2v = edge_num_naive(tet)
    lil = scipy.sparse.lil_matrix \
    ( (v2e.nnz, v2e.nnz), dtype=np.int64 )
    tet_e = v2e[tet[:,vp[0]], tet[:,vp[1]]].toarray()
    for v in tet_e:
        lil[v[:, None], v] = 1
    perm = scipy.sparse.csgraph.reverse_cuthill_mckee(lil.tocsr())
    e2v = e2v[perm]
    v2e = scipy.sparse.csr_matrix \
    ( ( np.arange(e2v.shape[0])
      , (e2v[:,0], e2v[:,1]) ) )
    bwh = 0
    for x in tet[:, vp]:
        y = v2e[x[0], x[1]]
        ymax = np.max(y)
        ymin = np.min(y)
        diff = ymax - ymin
        if bwh < diff:
            bwh = diff
    return v2e, bwh

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
    v2e, bwh = edge_num_banded(tet)
    pgroups = [('e', ([0,0,1],), lin), ('b', (), tri), ('v', (), tet)]
    for freq in [10e3, 100e3, 1e6]:
        sol = solve_geom(freq, vrt, pgroups, v2e.nnz, v2e, bwh)
        print(sol)
        for ptype, attr, nodes in pgroups:
            if ptype == 'e':
                print(isrc_v(sol, vrt, nodes, v2e, attr[0]))
        print(1/(e0*0.5*2*np.pi*freq))

if __name__ == '__main__':
    main()