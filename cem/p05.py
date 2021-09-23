import numpy as np
import scipy.sparse
import gmsh
import p01
import p04
import g03

def get_mesh():
    gmsh.initialize()
    air_tag, pec_tags, isrc_tag = g03.make_geom()
    gmsh.model.occ.synchronize()
    isrc, pec, air = g03.assign_physicals(air_tag, pec_tags, isrc_tag)
    nodes, elems = g03.gen_mesh()
    gmsh.finalize()
    ret_elems = []
    for e in elems:
        if e[0] == isrc:
            x = e[3][0].reshape(-1, 2) - 1
        elif e[0] == pec:
            x = e[3][0].reshape(-1, 3) - 1
        elif e[0] == air:
            x = e[3][0].reshape(-1, 4) - 1
        x.sort()
        ret_elems.append((e[0], x))
    return isrc, pec, air, nodes[1].reshape(-1,3), ret_elems

def main():
    np.set_printoptions(precision=3)
    ptisrc, ptpec, ptair, vrt, pgroups = get_mesh()
    tet = []
    for ptype, nodes in pgroups:
        if ptype == ptair:
            tet.append(nodes)
    tet = np.concatenate(tuple(tet))
    e2v = np.unique(tet[:, np.moveaxis(p04.vp,0,1)].reshape(-1,2), axis=0)
    v2e = scipy.sparse.csr_matrix \
    ( ( np.arange(e2v.shape[0])
      , (e2v[:,0], e2v[:,1]) ) )
    for freq in [10e3, 100e3, 1e6]:
        p04.solve_geom(freq, np.moveaxis(vrt,0,1), pgroups, e2v, v2e)

if __name__ == '__main__':
    main()