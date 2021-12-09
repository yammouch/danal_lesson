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

def local2global(glo, dst, loc, bwh): # inplace
    if bwh:
        glo[dst[...,np.newaxis] + bwh - dst, dst] += loc
    else:
        glo[dst[...,np.newaxis], dst] += loc

def racc(glo, ie1, val):
    glo[ie1] += val

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

def pec(glo, rhs, edge0, bwh):
    if bwh:
        for e in edge0:
            glo[ range(glo.shape[0])
               , np.arange(e+bwh, e-bwh-1, -1)%(glo.shape[1]) ] = 0
            glo[bwh, e] = 1
    else:
        glo[edge0       ] = 0
        glo[edge0, edge0] = 1
    rhs[edge0] = 0

def solve_geom(freq, vrt, pgroups, nedge, v2e, bwh):
    if bwh:
        lhs = np.zeros((2*bwh+1, nedge), dtype=np.complex128)
    else:
        lhs = np.zeros((nedge, nedge), dtype=np.complex128)
    rhs = np.zeros((nedge,), dtype=np.complex128)
    vas = {2: vl, 3: vs, 4: vp}
    for ptype, attr, nodes in pgroups:
        p2 = vrt[nodes]
        v = vas[nodes.shape[1]]
        ie2 = v2e[nodes[:,v[0]], nodes[:,v[1]]]
        if ptype == 'v': # air
            for p1, ie1 in zip(p2, ie2.toarray()):
                val = p01.volume(freq, p1, attr[0], attr[1], attr[2])
                local2global(lhs, ie1, val, bwh)
        elif ptype == 'b': # boundary condition
            for p1, ie1 in zip(p2, ie2.toarray()):
                val = None
                pec(lhs, rhs, ie1, bwh)
        elif ptype == 'a': # absorbing boundary
            for p1, ie1 in zip(p2, ie2.toarray()):
                val = p01.absorb(freq, p1)
                local2global(lhs, ie1, val, bwh)
        elif ptype == 'e': # excitation
            for p1, ie1 in zip(p2, ie2.toarray()):
                val = p01.isrc(freq, p1, attr[0])
                racc(rhs, ie1, val)
        elif ptype == 'e2': # excitation 2D
            for p1, ie1 in zip(p2, ie2.toarray()):
                val = p01.isrc_2d(freq, p1, attr[0])
                racc(rhs, ie1, val)
        elif ptype == 'e3': # excitation 3D
            for p1, ie1 in zip(p2, ie2.toarray()):
                val = p01.isrc_3d(freq, p1, attr[0])
                racc(rhs, ie1, val)
        elif ptype == 'p': # probe
            pass
        else:
            raise Exception("Unsupported physical type {}".format(ptype))
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
