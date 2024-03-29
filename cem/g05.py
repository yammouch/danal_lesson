import numpy as np
import scipy.sparse
import gmsh
import p01
import p04

def make_geom(size, isize):
    b = gmsh.model.occ.addBox(-size/2, -size/2, -size/2, size, size, size)
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

def get_mesh():
    gmsh.initialize()
    air_tag, pec_tags, isrc_tag = make_geom(1, 0.01)
    gmsh.model.occ.synchronize()
    isrc, pec, air = assign_physicals(air_tag, pec_tags, isrc_tag)
   #gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 3)
   #gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.7)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.1)
   #gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.model.mesh.generate(3)
    nodes = gmsh.model.mesh.getNodes()
    lacc = []
    racc = []
    probe = []
    for ntag in gmsh.model.getEntitiesForPhysicalGroup(1, isrc):
        es = gmsh.model.mesh.getElements(1, ntag)
        ns = es[2][0].reshape(-1, 2) - 1
        ns.sort()
        racc.append((p01.isrc(1, [0,0,1]), ns))
        probe.append(ns)
    for ntag in gmsh.model.getEntitiesForPhysicalGroup(2, pec):
        es = gmsh.model.mesh.getElements(2, ntag)
        ns = es[2][0].reshape(-1, 3) - 1
        ns.sort()
        lacc.append((p01.absorb, ns))
    for ntag in gmsh.model.getEntitiesForPhysicalGroup(3, air):
        es = gmsh.model.mesh.getElements(3, ntag)
        ns = es[2][0].reshape(-1, 4) - 1
        ns.sort()
        lacc.append((p01.volume(0, p04.e0, p04.u0), ns))
    gmsh.finalize()
    return nodes[1].reshape(-1,3), lacc, racc, probe


def main():
    np.set_printoptions(precision=3)
    vrt, lacc, racc, probe = get_mesh()
    solver = p04.Banded(vrt, lacc, racc, [])
    print(solver.v2e.nnz, solver.bwh)
    for freq in [1e9, 2e9]:
        sol = solver.solve(freq)
        print(sol)
        print(p04.isrc_v(sol, vrt, probe[0], solver.v2e, [0,0,1]))
        print(2*np.pi/3*(p04.u0/p04.e0)**0.5*(0.01/(p04.c/freq))**2)

if __name__ == '__main__':
    main()
