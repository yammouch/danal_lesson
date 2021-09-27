import numpy as np
import scipy.sparse
import gmsh
import p04
import g03

def get_mesh():
    gmsh.initialize()
    air_tag, pec_tags, isrc_tag = g03.make_geom()
    gmsh.model.occ.synchronize()
    isrc, pec, air = g03.assign_physicals(air_tag, pec_tags, isrc_tag)
   #gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 3)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.7)
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    nodes, elems = g03.gen_mesh()
    gmsh.finalize()
    ret_elems = []
    for e in elems:
        if e[0] == isrc:
            ptype = 'e'
            x = e[3][0].reshape(-1, 2) - 1
        elif e[0] == pec:
            ptype = 'b'
            x = e[3][0].reshape(-1, 3) - 1
        elif e[0] == air:
            ptype = 'v'
            x = e[3][0].reshape(-1, 4) - 1
        x.sort()
        ret_elems.append((ptype, (), x))
    return nodes[1].reshape(-1,3), ret_elems

def main():
    np.set_printoptions(precision=3)
    vrt, pgroups = get_mesh()
    tet = []
    for ptype, _, nodes in pgroups:
        if ptype == 'v':
            tet.append(nodes)
    print(pgroups)
    tet = np.concatenate(tuple(tet))
    e2v = np.unique(tet[:, np.moveaxis(p04.vp,0,1)].reshape(-1,2), axis=0)
    v2e = scipy.sparse.csr_matrix \
    ( ( np.arange(e2v.shape[0])
      , (e2v[:,0], e2v[:,1]) ) )
    print(v2e)
    for freq in [100, 1e3, 10e3, 100e3, 1e6]:
        p04.solve_geom(freq, np.moveaxis(vrt,0,1), pgroups, e2v, v2e)

if __name__ == '__main__':
    main()