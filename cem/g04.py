import numpy as np
import scipy.sparse
import gmsh
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
    air_tag, pec_tags, isrc_tag = make_geom(1, 1, 1)
    gmsh.model.occ.synchronize()
    isrc, pec, air = assign_physicals(air_tag, pec_tags, isrc_tag)
   #gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 3)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.7)
    gmsh.option.setNumber("Mesh.Algorithm", 8)
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
    tet = []
    for ptype, _, nodes in pgroups:
        if ptype == 'v':
            tet.append(nodes)
    print(pgroups)
    tet = np.concatenate(tuple(tet))
    v2e, bwh = p04.edge_num_banded(tet)
    print(v2e.nnz, bwh)
    for freq in [100, 1e3, 10e3, 100e3, 1e6]:
        p04.solve_geom \
        ( freq, np.moveaxis(vrt,0,1), pgroups, v2e.nnz, v2e, bwh )

if __name__ == '__main__':
    main()
