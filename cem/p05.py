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
        attr = ()
        if e[0] == isrc:
            ptype = 'e'
            x = e[3][0].reshape(-1, 2) - 1
            attr = ([0,0,1],)
        elif e[0] == pec:
            ptype = 'b'
            x = e[3][0].reshape(-1, 3) - 1
        elif e[0] == air:
            ptype = 'v'
            x = e[3][0].reshape(-1, 4) - 1
        x.sort()
        ret_elems.append((ptype, attr, x))
    return nodes[1].reshape(-1,3), ret_elems

def main():
    np.set_printoptions(precision=3)
    vrt, pgroups = get_mesh()
    vrt = np.moveaxis(vrt, 0, 1)
    tet = []
    for ptype, _, nodes in pgroups:
        if ptype == 'v':
            tet.append(nodes)
    print(pgroups)
    tet = np.concatenate(tuple(tet))
    v2e, bwh = p04.edge_num_banded(tet)
    for freq in [100, 1e3, 10e3, 100e3, 1e6]:
        sol = p04.solve_geom(freq, vrt, pgroups, v2e.nnz, v2e, bwh)
        print(sol)
        for ptype, attr, nodes in pgroups:
            if ptype == 'e':
                print(p04.isrc_v(sol, vrt, nodes, v2e, attr[0]))
        print(1/(p04.e0*0.5*2*np.pi*freq))

if __name__ == '__main__':
    main()