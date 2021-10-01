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
  return air, cond[0][0][1], isrc

def assign_physicals(air_tag, cond_tag, isrc_tag):
  isrc = gmsh.model.addPhysicalGroup(1, [isrc_tag])
  cond = gmsh.model.addPhysicalGroup(2, [cond_tag])
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
        if e[0] == isrc:
            ptype = 'e'
            x = e[3][0].reshape(-1, 2) - 1
        elif e[0] == pec:
            ptype = 'a'
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
    tet = np.concatenate(tuple(tet))
    v2e, bwh = p04.edge_num_banded(tet)
    print(v2e.nnz, bwh)
    for freq in [1e9, 2e9]:
        p04.solve_geom \
        ( freq, np.moveaxis(vrt,0,1), pgroups, v2e.nnz, v2e, bwh )

if __name__ == '__main__':
    main()
