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

def volume(glo, freq, vrt, tet, attr, v2e, bwh):
    sigma   = attr[0]
    epsilon = attr[1]
    mu      = attr[2]
    for t in tet:
        n, vol = p01.ntet(vrt[t])
        stiff = p01.make_stiff(n, vol)
        mass = p01.make_mass(n, vol)
        w = 2*np.pi*freq
        local2global(glo, t, v2e, stiff/mu, bwh)
        local2global(glo, t, v2e, -w*(w*epsilon-1j*sigma)*mass, bwh)

def absorb(lhs, freq, vrt, nodes, v2e, bwh):
    for t in nodes:
        n, area = p01.ntri2(vrt[t])
        ab = p01.bound(n, area)
        local2global2d(lhs, t, v2e, 2j*np.pi*freq*np.sqrt(e0/u0)*ab, bwh)

def isrc1(v, nfn, freq, vrt, nodes, v2e, i_density):
    n, jacob = nfn(vrt[nodes])
    x  = n[v[1]] - n[v[0]]
    x *= np.array(i_density)
    x  = x.sum(axis=-1)
    x *= np.abs(jacob)
    x = -2j*np.pi*freq*x
    return x

def isrc(rhs, freq, vrt, nodes, v2e, i_density):
    for i in nodes:
        x = isrc1(vl, lambda p: (p[[1, 0]] - p[[0, 1]], 1.0), freq, vrt, i, v2e, i_density)
        dst = v2e[tuple(i[vl])]
        rhs[dst[0]] += x/2

def isrc_2d(rhs, freq, vrt, nodes, v2e, i_density):
    for i in nodes:
        x = isrc1(vs, p01.ntri2, freq, vrt, i, v2e, i_density)
        dst = v2e[tuple(i[vs])]
        rhs[dst[0]] += x/6

def isrc_3d(rhs, freq, vrt, nodes, v2e, i_density):
    for i in nodes:
        x = isrc1(vp, p01.ntet, freq, vrt, i, v2e, i_density)
        dst = v2e[tuple(i[vp])]
        rhs[dst[0]] += x/24

def solve_geom(freq, vrt, pgroups, nedge, v2e, bwh):
    if bwh:
        lhs = np.zeros((2*bwh+1, nedge), dtype=np.complex128)
    else:
        lhs = np.zeros((nedge, nedge), dtype=np.complex128)
    rhs = np.zeros((nedge,), dtype=np.complex128)
    dirichlet = []
    for ptype, attr, nodes in pgroups:
        if ptype == 'v': # air
            volume(lhs, freq, vrt, nodes, attr, v2e, bwh)
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
