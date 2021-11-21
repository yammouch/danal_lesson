import numpy as np
import scipy.sparse
import gmsh
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
    air_tag, cond_tag, isrc_tag = make_geom()
    print(air_tag, cond_tag, isrc_tag)
    gmsh.model.occ.synchronize()
    isrc, cond, air = assign_physicals(air_tag, cond_tag, isrc_tag)
    nodes, elems = gen_mesh()
    gmsh.finalize()
    ret_elems = []
    for e in elems:
        attr = ()
        if e[0] == isrc:
            ptype = 'e'
            x = e[3][0].reshape(-1, 2) - 1
            attr = ([0,1,0],)
        elif e[0] == cond:
            ptype = 'v'
            x = e[3][0].reshape(-1, 4) - 1
            attr = (1/140e-8, p04.e0, p04.u0)
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
    vrt *= 1e-3 # [m] -> [mm]
    vrt = np.moveaxis(vrt, 0, 1)
    tet = []
    for ptype, _, nodes in pgroups:
        if ptype in ['v', 'c']:
            tet.append(nodes)
    tet = np.concatenate(tuple(tet))
    v2e, bwh = p04.edge_num_banded(tet)
    print(v2e.nnz, bwh)
    for freq in [1, 10, 100, 1e3, 10e3, 100e6]:
        sol = p04.solve_geom(freq, vrt, pgroups, v2e.nnz, v2e, bwh)
        print(sol)
        for ptype, attr, nodes in pgroups:
            if ptype == 'e':
                print(p04.isrc_v(sol, vrt, nodes, v2e, attr[0]))
        print(140e-8*2*np.pi*0.1/(0.01**2*np.pi))
        print(p04.u0*0.1*(np.log(8*0.1/0.01)-2)*2*np.pi*freq)
        # https://www.emisoftware.com/calculator/circular-loop/

if __name__ == '__main__':
    main()
