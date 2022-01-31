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

def lacc(lhs, _, ie1, val, bwh): # inplace
    if bwh:
        lhs[ie1[...,np.newaxis] + bwh - ie1, ie1] += val
    else:
        lhs[ie1[...,np.newaxis], ie1] += val

def racc(_0, rhs, ie1, val, _1):
    rhs[ie1] += val

def isrc_v(sol, vrt, nodes, v2e, isrc_dir):
    diff = vrt[nodes[:,1]] - vrt[nodes[:,0]]
    mag = (diff**2).sum(axis=-1)**0.5
    dirc = diff/mag[:,None]
    ip = (dirc*isrc_dir).sum(axis=-1)
    return (sol[v2e[nodes[:,0],nodes[:,1]][0]] * ip).sum()

def pec(lhs, rhs, edge0, _, bwh):
    if bwh:
        for e in edge0:
            lhs[ range(lhs.shape[0])
               , np.arange(e+bwh, e-bwh-1, -1)%(lhs.shape[1]) ] = 0
            lhs[bwh, e] = 1
    else:
        lhs[edge0       ] = 0
        lhs[edge0, edge0] = 1
    rhs[edge0] = 0

class Square(object):

    def __init__(self, vrt, pgroups):
        super().__init__()
        self.vrt = vrt
        self.pgroups = pgroups
        tet = []
        for pg in pgroups:
            if pg[2].shape[1] == 4:
                tet.append(pg[2])
        tet = np.concatenate(tuple(tet))
        self.v2e, self.bwh = edge_num_banded(tet)
        self.nedge = self.v2e.nnz

    def solve(self, freq):
        if self.bwh:
            lhs = np.zeros((2*self.bwh+1, self.nedge), dtype=np.complex128)
        else:
            lhs = np.zeros((self.nedge, self.nedge), dtype=np.complex128)
        rhs = np.zeros((self.nedge,), dtype=np.complex128)
        vas = {2: vl, 3: vs, 4: vp}
        for ptype, attr, nodes in self.pgroups:
            p2 = self.vrt[nodes]
            v = vas[nodes.shape[1]]
            ie2 = self.v2e[nodes[:,v[0]], nodes[:,v[1]]]
            for p1, ie1 in zip(p2, ie2.toarray()):
                val = attr(freq, p1)
                ptype(lhs, rhs, ie1, val, self.bwh)
        if self.bwh:
            sol = scipy.linalg.solve_banded \
            ( (self.bwh, self.bwh), lhs, rhs, overwrite_ab=True, overwrite_b=True )
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
