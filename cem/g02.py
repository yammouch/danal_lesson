import gmsh

gmsh.initialize()

p = [gmsh.model.occ.addPoint(*a) for a in
     [ (0,0,-1), (2**0.5,-1,0), (2**0.5,1,0), (0,0,1)
     , (0.5, 0, -0.2), (0.5, 0, 0.2)] ]
l = [gmsh.model.occ.addLine(*[p[i] for i in a])
     for a in [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3), (4,5)] ]
c = [gmsh.model.occ.addCurveLoop([l[i] for i in a])
     for a in [(0,1,3), (0,2,4), (1,2,5), (3,4,5)] ]
s = [gmsh.model.occ.addPlaneSurface([i]) for i in c]
sl = gmsh.model.occ.addSurfaceLoop(s)
v = gmsh.model.occ.addVolume([sl])
f = gmsh.model.occ.fragment([(3, v), (1, l[6])], [])
print(f)

gmsh.model.occ.synchronize()
isrc = gmsh.model.addPhysicalGroup(1, [f[1][1][0][1]])
radi = gmsh.model.addPhysicalGroup(2, [l[1], l[2]])
pec  = gmsh.model.addPhysicalGroup(2, [l[0], l[3]])
vg = gmsh.model.addPhysicalGroup(3, [f[1][0][0][1]])
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 3)
#gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 1.5)
gmsh.model.mesh.generate(3)
print(gmsh.model.mesh.getNodes())
for dim, ptag in gmsh.model.getPhysicalGroups():
  for ntag in gmsh.model.getEntitiesForPhysicalGroup(dim, ptag):
    print(gmsh.model.mesh.getElements(dim, ntag))
#gmsh.write("g02.msh2")
gmsh.fltk.run()
