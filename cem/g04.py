import numpy as np
import scipy.sparse
import gmsh
import p01
import p04

def make_geom(w, l, h):
    b = gmsh.model.occ.addBox(-w/2, 0, 0, w, l, h)
    lower = gmsh.model.occ.addRectangle(-w/2, 0, 0, w, l)
    upper = gmsh.model.occ.addRectangle(-w/2, 0, h, w, l)
    vert  = gmsh.model.occ.addRectangle(-w/2, 0, 0, w, h)
    gmsh.model.occ.rotate   ([(2, vert)], 0, 0, 0, 1, 0, 0, np.pi/2)
    gmsh.model.occ.translate([(2, vert)], 0, l, 0)
    isrcl = gmsh.model.occ.addPoint(0, 0, 0)
    isrcu = gmsh.model.occ.addPoint(0, 0, h)
    isrc  = gmsh.model.occ.addLine(isrcl, isrcu)
    f = gmsh.model.occ.fragment \
    ( [ (3, b), (2, lower), (2, upper), (2, vert), (1, isrc) ]
    , [] )
    print(f)
    return b, (lower, upper, vert), isrc

def assign_physicals(air_tag, pec_tags, isrc_tag):
    isrc = gmsh.model.addPhysicalGroup(1, [isrc_tag])
    pec  = gmsh.model.addPhysicalGroup(2, pec_tags)
    air  = gmsh.model.addPhysicalGroup(3, [air_tag])
    return isrc, pec, air

def get_mesh(w, l, h):
    gmsh.initialize()
    air_tag, pec_tags, isrc_tag = make_geom(w, l, h)
    gmsh.model.occ.synchronize()
    isrc, pec, air = assign_physicals(air_tag, pec_tags, isrc_tag)
   #gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 3)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.4)
   #gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.model.mesh.generate(3)
    nodes = gmsh.model.mesh.getNodes()
    tet = []
    ret_elems = []
    probe = []
    for ntag in gmsh.model.getEntitiesForPhysicalGroup(1, isrc):
        es = gmsh.model.mesh.getElements(1, ntag)
        ns = es[2][0].reshape(-1, 2) - 1
        ns.sort()
        ret_elems.append \
        ( ( p04.racc
          , p01.isrc(1, [0,0,1])
          , ns ) )
        probe.append(ns)
    for ntag in gmsh.model.getEntitiesForPhysicalGroup(2, pec):
        es = gmsh.model.mesh.getElements(2, ntag)
        ns = es[2][0].reshape(-1, 3) - 1
        ns.sort()
        ret_elems.append \
        ( ( p04.pec
          , lambda f, p: None
          , ns ) )
    for ntag in gmsh.model.getEntitiesForPhysicalGroup(3, air):
        es = gmsh.model.mesh.getElements(3, ntag)
        ns = es[2][0].reshape(-1, 4) - 1
        ns.sort()
        ret_elems.append \
        ( ( p04.lacc
          , p01.volume(0, p04.e0, p04.u0)
          , ns ) )
        tet.append(ns)
    gmsh.finalize()
    return nodes[1].reshape(-1,3), ret_elems, tet, probe

def main():
    w, l, h = 1, 10, 1
    np.set_printoptions(precision=3)
    vrt, pgroups, tet, probe = get_mesh(w, l, h)
    print(pgroups)
    tet = np.concatenate(tuple(tet))
    v2e, bwh = p04.edge_num_banded(tet)
    print(v2e.nnz, bwh)
    for freq in [100, 1e3, 10e3, 100e3, 1e6]:
        sol = p04.solve_geom(freq, vrt, pgroups, v2e.nnz, v2e, bwh)
        print(sol)
        print(p04.isrc_v(sol, vrt, probe[0], v2e, [0,0,1]))
        print(2*np.pi*freq*p04.u0*(l*h)/w)

if __name__ == '__main__':
    main()
