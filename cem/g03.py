import gmsh

def make_geom():
  p = [gmsh.model.occ.addPoint(*a) for a in
       [ (0,0,0), (1,0,0), (0,1,0)
       , (0,0,1), (1,0,1), (0,1,1) ] ]
  print(p)
  l = [gmsh.model.occ.addLine(*[p[i] for i in a])
       for a in [(0,1), (1,2), (2,0), (3,4), (4,5), (5,3), (0,3)] ]
  print(l)
  c = [gmsh.model.occ.addCurveLoop([l[i] for i in a])
       for a in [(0,1,2), (3,4,5)] ]
  print(c)
  s = [gmsh.model.occ.addPlaneSurface([i]) for i in c+[c[0]] ]
  print(s)
  v = gmsh.model.occ.extrude([(2,s[2])], 0, 0, 1)
  print(v)
  f = gmsh.model.occ.fragment([v[1], (2,s[0]), (2,s[1]), (1,l[6])], [])
  print(f)
  return v[1][1], (s[0], s[1]), l[6]

def assign_physicals(air_tag, pec_tags, isrc_tag):
  isrc = gmsh.model.addPhysicalGroup(1, [isrc_tag])
  pec  = gmsh.model.addPhysicalGroup(2, pec_tags)
  air  = gmsh.model.addPhysicalGroup(3, [air_tag])
  return isrc, pec, air

def gen_mesh():
  gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 3)
  gmsh.model.mesh.generate(3)
  nodes = gmsh.model.mesh.getNodes()
  elems = []
  for dim, ptag in gmsh.model.getPhysicalGroups():
    for ntag in gmsh.model.getEntitiesForPhysicalGroup(dim, ptag):
      elems.append(gmsh.model.mesh.getElements(dim, ntag))
  return nodes, elems