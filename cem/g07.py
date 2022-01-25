import numpy as np
import scipy.sparse
import gmsh
import p01
import p04

def make_geom():
    asize = 400 # [mm]
    r1 = 100
    r2 = 10
    d = 1
    air = gmsh.model.occ.addBox \
    ( -asize/2, -asize/2, -asize/2, asize, asize, asize )
    cond = gmsh.model.occ.addTorus(0, 0, 0, r1, r2)
    cond_cut = gmsh.model.occ.addBox(0, -d/2, -r2, asize/2, d, 2*r2)
    cond = gmsh.model.occ.cut([(3, cond)], [(3, cond_cut)])
    print(cond)
    isrcl = gmsh.model.occ.addPoint(r1, -d/2, 0)
    isrcu = gmsh.model.occ.addPoint(r1,  d/2, 0)
    isrc  = gmsh.model.occ.addLine(isrcl, isrcu)
    f = gmsh.model.occ.fragment \
    ( [ (3, air), cond[0][0], (1, isrc) ]
    , [] )
    print(f)
    return f[1][0][0][1], cond[0][0][1], isrc

def assign_physicals(air_tag, cond_tag, isrc_tag):
    isrc = gmsh.model.addPhysicalGroup(1, [isrc_tag])
    cond = gmsh.model.addPhysicalGroup(3, [cond_tag])
    air  = gmsh.model.addPhysicalGroup(3, [air_tag])
    return isrc, cond, air

def get_mesh():
    gmsh.initialize()
    air_tag, cond_tag, isrc_tag = make_geom()
    print(air_tag, cond_tag, isrc_tag)
    gmsh.model.occ.synchronize()
    isrc, cond, air = assign_physicals(air_tag, cond_tag, isrc_tag)
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
          , p01.isrc(1, [0,1,0])
          , ns ) )
        probe.append(ns)
    for ntag in gmsh.model.getEntitiesForPhysicalGroup(3, cond):
        es = gmsh.model.mesh.getElements(3, ntag)
        ns = es[2][0].reshape(-1, 4) - 1
        ns.sort()
        ret_elems.append \
        ( ( p04.lacc
          , p01.volume(1/140e-8, p04.e0, p04.u0)
          , ns ) )
        tet.append(ns)
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
    np.set_printoptions(precision=3)
    vrt, pgroups, tet, probe = get_mesh()
    vrt *= 1e-3 # [m] -> [mm]
    tet = np.concatenate(tuple(tet))
    v2e, bwh = p04.edge_num_banded(tet)
    print(v2e.nnz, bwh)
    for freq in [1, 10, 100, 1e3, 10e3, 100e6]:
        sol = p04.solve_geom(freq, vrt, pgroups, v2e.nnz, v2e, bwh)
        print(sol)
        print(p04.isrc_v(sol, vrt, probe[0], v2e, [0,1,0]))
        print(140e-8*2*np.pi*0.1/(0.01**2*np.pi))
        print(p04.u0*0.1*(np.log(8*0.1/0.01)-2)*2*np.pi*freq)
        # https://www.emisoftware.com/calculator/circular-loop/

if __name__ == '__main__':
    main()
