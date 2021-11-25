import numpy as np
import scipy.sparse
import gmsh
import p04

def make_geom(size, isize):
    b = gmsh.model.occ.addSphere(0, 0, 0, size/2)
    isrcl = gmsh.model.occ.addPoint(0, 0, -isize/2)
    isrcu = gmsh.model.occ.addPoint(0, 0,  isize/2)
    isrc  = gmsh.model.occ.addLine(isrcl, isrcu)
    f = gmsh.model.occ.fragment \
    ( [ (3, b), (1, isrc) ]
    , [] )
    print(f)
    gmsh.model.occ.synchronize()
    bd = gmsh.model.getBoundary([(3, b)])
    return b, [x[1] for x in bd], isrc

def assign_physicals(air_tag, pec_tags, isrc_tag):
    isrc = gmsh.model.addPhysicalGroup(1, [isrc_tag])
    pec  = gmsh.model.addPhysicalGroup(2, pec_tags)
    air  = gmsh.model.addPhysicalGroup(3, [air_tag])
    return isrc, pec, air

def gen_mesh():
    gmsh.model.mesh.generate(3)
    nodes = gmsh.model.mesh.getNodes()
    elems = []
    for dim, ptag in gmsh.model.getPhysicalGroups():
        for ntag in gmsh.model.getEntitiesForPhysicalGroup(dim, ptag):
            elems.append((ptag,) + gmsh.model.mesh.getElements(dim, ntag))
    return nodes, elems

def get_mesh():
    gmsh.initialize()
    air_tag, pec_tags, isrc_tag = make_geom(1, 0.01)
    gmsh.model.occ.synchronize()
    isrc, pec, air = assign_physicals(air_tag, pec_tags, isrc_tag)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.1)
   #gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.7)
   #gmsh.option.setNumber("Mesh.Algorithm", 8)
    nodes, elems = gen_mesh()
    gmsh.finalize()
    ret_elems = []
    for e in elems:
        attr = ()
        if e[0] == isrc:
            ptype = 'e'
            x = e[3][0].reshape(-1, 2) - 1
            attr = ([0,0,1],)
        elif e[0] == pec:
            ptype = 'a'
            x = e[3][0].reshape(-1, 3) - 1
        elif e[0] == air:
            ptype = 'v'
            x = e[3][0].reshape(-1, 4) - 1
            attr = (0, p04.e0, p04.u0)
        x.sort()
        ret_elems.append((ptype, attr, x))
    return nodes[1].reshape(-1,3), ret_elems

def main():
    np.set_printoptions(precision=3)
    vrt, pgroups = get_mesh()
    tet = []
    for ptype, _, nodes in pgroups:
        if ptype == 'v':
            tet.append(nodes)
    tet = np.concatenate(tuple(tet))
    v2e, bwh = p04.edge_num_banded(tet)
    print(v2e.nnz, bwh)
    for freq in [1e9, 2e9]:
        sol = p04.solve_geom(freq, vrt, pgroups, v2e.nnz, v2e, bwh)
        print(sol)
        for ptype, attr, nodes in pgroups:
            if ptype == 'e':
                print(p04.isrc_v(sol, vrt, nodes, v2e, attr[0]))
        print(2*np.pi/3*(p04.u0/p04.e0)**0.5*(0.01/(p04.c/freq))**2)

if __name__ == '__main__':
    main()
