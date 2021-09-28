import numpy as np
import scipy.sparse
import gmsh
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
    air_tag, pec_tags, isrc_tag = make_geom(3, 0.001)
    gmsh.model.occ.synchronize()
    isrc, pec, air = assign_physicals(air_tag, pec_tags, isrc_tag)
   #gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 3)
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
    e2v = np.unique(tet[:, np.moveaxis(p04.vp,0,1)].reshape(-1,2), axis=0)
    v2e = scipy.sparse.csr_matrix \
    ( ( np.arange(e2v.shape[0])
      , (e2v[:,0], e2v[:,1]) ) )
    for freq in [1e9, 2e9, 5e9, 10e9, 20e9]:
        p04.solve_geom(freq, np.moveaxis(vrt,0,1), pgroups, e2v, v2e)

if __name__ == '__main__':
    main()
