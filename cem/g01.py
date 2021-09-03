import gmsh

gmsh.initialize()

p = [gmsh.model.occ.addPoint(*a)
     for a in [(0,0,-1), (0,0,1), (2**0.5,-1,0), (2**0.5,1,0)] ]
l = [gmsh.model.occ.addLine(*[p[i] for i in a])
     for a in [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)] ]
c = [gmsh.model.occ.addCurveLoop([l[i] for i in a])
     for a in [(0,1,3), (0,2,4), (1,2,5), (3,4,5)] ]
s = [gmsh.model.occ.addPlaneSurface([i]) for i in c]
sl = gmsh.model.occ.addSurfaceLoop(s)
v = gmsh.model.occ.addVolume([sl])

gmsh.model.occ.synchronize()
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 3)
gmsh.model.mesh.generate(3)
#gmsh.write("g01.msh2")
gmsh.fltk.run()
