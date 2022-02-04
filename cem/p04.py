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

    def __init__(self, vrt, lacc, racc, pec):
        self.vrt = vrt
        self.lacc = lacc
        self.racc = racc
        self.pec = pec
        tet = []
        for pg in sum([racc, lacc, pec], []):
            if pg[1].shape[1] == 4:
                tet.append(pg[1])
        tet = np.concatenate(tuple(tet))
        self.v2e, self.bwh = edge_num_naive(tet)
        self.nedge = self.v2e.nnz
        self.lhs = np.zeros((self.nedge, self.nedge), dtype=np.complex128)
        self.rhs = np.empty((self.nedge,), dtype=np.complex128)

    def f_lacc(self, ie1, val):
        self.lhs[ie1[...,np.newaxis], ie1] += val

    def f_racc(self, ie1, val):
        self.rhs[ie1] += val

    def f_pec(self, edge0, _):
        self.lhs[edge0       ] = 0
        self.lhs[edge0, edge0] = 1
        self.rhs[edge0] = 0

    def call_lin(self):
        sol = np.linalg.solve(self.lhs, self.rhs)
        return sol

    def solve(self, freq):
        self.lhs[:] = 0
        self.rhs[:] = 0
        vas = {2: vl, 3: vs, 4: vp}
        for f, l in [ (self.f_lacc, self.lacc)
                    , (self.f_racc, self.racc)
                    , (self.f_pec, self.pec)]:
            for attr, nodes in l:
                p2 = self.vrt[nodes]
                v = vas[nodes.shape[1]]
                ie2 = self.v2e[nodes[:,v[0]], nodes[:,v[1]]]
                for p1, ie1 in zip(p2, ie2.toarray()):
                    val = attr(freq, p1)
                    f(ie1, val)
        sol = self.call_lin()
        return sol


class Banded(Square):

    def __init__(self, vrt, lacc, racc, pec):
        self.vrt = vrt
        self.lacc = lacc
        self.racc = racc
        self.pec = pec
        tet = []
        for pg in sum([racc, lacc, pec], []):
            if pg[1].shape[1] == 4:
                tet.append(pg[1])
        tet = np.concatenate(tuple(tet))
        self.v2e, self.bwh = edge_num_banded(tet)
        self.nedge = self.v2e.nnz
        self.lhs = np.empty((2*self.bwh+1, self.nedge), dtype=np.complex128)
        self.rhs = np.empty((self.nedge,), dtype=np.complex128)

    def f_lacc(self, ie1, val):
        self.lhs[ie1[...,np.newaxis] + self.bwh - ie1, ie1] += val

    def f_pec(self, edge0, _):
        for e in edge0:
            self.lhs[ range(self.lhs.shape[0])
                    , np.arange(e+self.bwh, e-self.bwh-1, -1)%(self.lhs.shape[1]) ] = 0
            self.lhs[self.bwh, e] = 1
        self.rhs[edge0] = 0

    def call_lin(self):
        sol = scipy.linalg.solve_banded \
        ( (self.bwh, self.bwh), self.lhs, self.rhs
        , overwrite_ab=True, overwrite_b=True )
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
