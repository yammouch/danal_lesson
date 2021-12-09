import numpy as np
import scipy.sparse
import scipy.linalg
import p01
import datetime

c  = 299792458.0
u0 = 4e-7*np.pi
e0 = 1/(u0*c**2)

vp = np.array([[0,0,0,1,1,2],[1,2,3,2,3,3]])
vs = np.array([[0,0,1],[1,2,2]])
vl = np.array([[0],[1]])

def local2global(glo, tet, v2e, loc, bwh): # inplace
    dst = v2e[tuple(tet[vp])]
    if bwh:
        glo[dst.T + bwh - dst, dst] += loc
    else:
        glo[dst.T, dst] += loc

def local2global_new(glo, dst, loc, bwh): # inplace
    if bwh:
        glo[dst[...,np.newaxis] + bwh - dst, dst] += loc
    else:
        glo[dst[...,np.newaxis], dst] += loc

def local2global2d(glo, tri, v2e, loc, bwh): # inplace
    dst = v2e[tuple(tri[vs])]
    if bwh:
        glo[dst.T + bwh - dst, dst] += loc
    else:
        glo[dst.T, dst] += loc

#def isrc(rhs, freq, vrt, nodes, v2e, isrc_dir):
#    diff = vrt[nodes[:,1]] - vrt[nodes[:,0]]
#    mag = (diff**2).sum(axis=-1)**0.5
#    dirc = diff/mag[:,None]
#    ip = (dirc*isrc_dir).sum(axis=-1)
#    rhs[v2e[nodes[:,0],nodes[:,1]]] = -2j*np.pi*freq*ip

def isrc_v(sol, vrt, nodes, v2e, isrc_dir):
    diff = vrt[nodes[:,1]] - vrt[nodes[:,0]]
    mag = (diff**2).sum(axis=-1)**0.5
    dirc = diff/mag[:,None]
    ip = (dirc*isrc_dir).sum(axis=-1)
    return (sol[v2e[nodes[:,0],nodes[:,1]][0]] * ip).sum()

def pec(glo, rhs, ie2, bwh):
    edge0 = ie2.toarray().reshape(-1)
    if bwh:
        for e in edge0:
            glo[ range(glo.shape[0])
               , np.arange(e+bwh, e-bwh-1, -1)%(glo.shape[1]) ] = 0
            glo[bwh, e] = 1
    else:
        glo[edge0       ] = 0
        glo[edge0, edge0] = 1
    rhs[edge0] = 0

def volume(glo, freq, p2, ie2, attr, bwh):
    sigma   = attr[0]
    epsilon = attr[1]
    mu      = attr[2]
    for p1, ie1 in zip(p2, ie2.toarray()):
        n, vol = p01.ntet(p1)
        stiff = p01.make_stiff(n, vol)
        mass = p01.make_mass(n, vol)
        w = 2*np.pi*freq
        local2global_new(glo, ie1, stiff/mu, bwh)
        local2global_new(glo, ie1, -w*(w*epsilon-1j*sigma)*mass, bwh)

def absorb(lhs, freq, p2, ie2, bwh):
    for p1, ie1 in zip(p2, ie2.toarray()):
        n, area = p01.ntri2(p1)
        ab = p01.bound(n, area)
        local2global_new(lhs, ie1, 2j*np.pi*freq*np.sqrt(e0/u0)*ab, bwh)

def isrc1(v, nfn, freq, p1, i_density):
    n, jacob = nfn(p1)
    x  = n[v[1]] - n[v[0]]
    x *= np.array(i_density)
    x  = x.sum(axis=-1)
    x *= np.abs(jacob)
    x = -2j*np.pi*freq*x
    return x

def isrc(rhs, freq, p2, ie2, i_density):
    for p1, ie1 in zip(p2, ie2.toarray()):
        x = isrc1(vl, lambda p: (p[[1, 0]] - p[[0, 1]], 1.0), freq, p1, i_density)
        rhs[ie1] += x/2

def isrc_2d(rhs, freq, p2, ie2, i_density):
    for p1, ie1 in zip(p2, ie2.toarray()):
        x = isrc1(vs, p01.ntri2, freq, p1, i_density)
        rhs[ie1] += x/6

def isrc_3d(rhs, freq, p2, ie2, i_density):
    for p1, ie1 in zip(p2, ie2.toarray()):
        x = isrc1(vp, p01.ntet, freq, p1, i_density)
        rhs[ie1] += x/24

def solve_geom(freq, vrt, pgroups, nedge, v2e, bwh):
    if bwh:
        lhs = np.zeros((2*bwh+1, nedge), dtype=np.complex128)
    else:
        lhs = np.zeros((nedge, nedge), dtype=np.complex128)
    rhs = np.zeros((nedge,), dtype=np.complex128)
    dirichlet = []
    for ptype, attr, nodes in pgroups:
        p2 = vrt[nodes]
        if ptype == 'v': # air
            ie2 = v2e[nodes[:,vp[0]], nodes[:,vp[1]]]
            volume(lhs, freq, p2, ie2, attr, bwh)
        elif ptype == 'b': # boundary condition
            dirichlet.append((attr, nodes))
        elif ptype == 'a': # absorbing boundary
            ie2 = v2e[nodes[:,vs[0]], nodes[:,vs[1]]]
            absorb(lhs, freq, p2, ie2, bwh)
        elif ptype == 'e': # excitation
            ie2 = v2e[nodes[:,vl[0]], nodes[:,vl[1]]]
            isrc(rhs, freq, p2, ie2, attr[0])
        elif ptype == 'e2': # excitation 2D
            ie2 = v2e[nodes[:,vs[0]], nodes[:,vs[1]]]
            isrc_2d(rhs, freq, p2, ie2, attr[0])
        elif ptype == 'e3': # excitation 3D
            ie2 = v2e[nodes[:,vp[0]], nodes[:,vp[1]]]
            isrc_3d(rhs, freq, p2, ie2, attr[0])
        elif ptype == 'p': # probe
            pass
        else:
            raise Exception("Unsupported physical type {}".format(ptype))
    for _, nodes in dirichlet:
        ie2 = v2e[nodes[:,vs[0]], nodes[:,vs[1]]]
        pec(lhs, rhs, ie2, bwh)
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
