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
   #print(gmsh.model.getBoundary([(3, air)]))
   #print(gmsh.model.getAdjacencies(3, air))
    torus = gmsh.model.occ.addTorus(0, 0, 0, r1, r2)
    print(torus)
    cond_cut = gmsh.model.occ.addBox(0, -d/2, -r2, asize/2, d, 2*r2)
    print(cond_cut)
    cond = gmsh.model.occ.cut([(3, torus)], [(3, cond_cut)], removeObject=False, removeTool=False)
    print(cond)
    isrc = gmsh.model.occ.intersect([(3, torus)], [(3, cond_cut)])
    print(isrc)
    probel = gmsh.model.occ.addPoint(r1, -d/2, 0)
    probeu = gmsh.model.occ.addPoint(r1,  d/2, 0)
    probe  = gmsh.model.occ.addLine(probel, probeu)
    f = gmsh.model.occ.fragment \
    ( [ (3, air), cond[0][0], isrc[0][0], (1, probe) ]
    , [] )
    gmsh.model.occ.synchronize()
    print(gmsh.model.getBoundary([(3, cond[0][0][1])], recursive=True))
    gmsh.model.mesh.setSize \
    ( gmsh.model.getBoundary([(3, cond[0][0][1])], recursive=True)
    , 12)
    print(f)
    return f[1][0][0][1], cond[0][0][1], isrc[0][0][1], probe

def assign_physicals(air_tag, cond_tag, isrc_tag, probe_tag):
    print(isrc_tag)
    isrc  = gmsh.model.addPhysicalGroup(3, [isrc_tag])
    probe = gmsh.model.addPhysicalGroup(1, [probe_tag])
    cond  = gmsh.model.addPhysicalGroup(3, [cond_tag])
    air   = gmsh.model.addPhysicalGroup(3, [air_tag])
    return probe, isrc, cond, air

def get_mesh():
    gmsh.initialize()
    air_tag, cond_tag, isrc_tag, probe_tag = make_geom()
    print(probe_tag, air_tag, cond_tag, isrc_tag)
    gmsh.model.occ.synchronize()
    probe, isrc, cond, air = assign_physicals(air_tag, cond_tag, isrc_tag, probe_tag)
   #gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 10)
   #gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.model.mesh.generate(3)
    nodes = gmsh.model.mesh.getNodes()
    gmsh.write('g09.msh2')
    tet = []
    ret_elems = []
    ret_probe = []
    for ntag in gmsh.model.getEntitiesForPhysicalGroup(3, isrc):
        es = gmsh.model.mesh.getElements(3, ntag)
        ns = es[2][0].reshape(-1, 4) - 1
        ns.sort()
        ret_elems.append \
        ( ( p04.racc
          , p01.isrc(3, [0,1/(0.01**2*np.pi),0])
          , ns ) )
        ret_elems.append \
        ( ( p04.lacc
          , p01.volume(0, p04.e0, p04.u0)
          , ns ) )
        tet.append(ns)
    for ntag in gmsh.model.getEntitiesForPhysicalGroup(1, probe):
        es = gmsh.model.mesh.getElements(1, ntag)
        ns = es[2][0].reshape(-1, 2) - 1
        ns.sort()
        ret_probe.append(ns)
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
    return nodes[1].reshape(-1,3), ret_elems, tet, ret_probe

def main():
    np.set_printoptions(precision=3)
    vrt, pgroups, tet, probe = get_mesh()
    vrt *= 1e-3 # [m] -> [mm]
    solver = p04.Square(vrt, pgroups)
    print(solver.v2e.nnz, solver.bwh)
   #for freq in [1, 10, 100, 1e3, 10e3, 100e6]:
    for freq in [1e3]:
   #for freq in []:
        sol = solver.solve(freq)
        print(sol)
        print(p04.isrc_v(sol, vrt, probe[0], solver.v2e, [0,1,0]))
        print(140e-8*2*np.pi*0.1/(0.01**2*np.pi))
        print(p04.u0*0.1*(np.log(8*0.1/0.01)-2)*2*np.pi*freq)
        # https://www.emisoftware.com/calculator/circular-loop/

if __name__ == '__main__':
    main()
