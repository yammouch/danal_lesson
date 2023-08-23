import gmsh

gmsh.initialize()

p = [gmsh.model.occ.addPoint(x, y, 0, 3) for x, y in [[0, -1], [0, 1], [2, 0]]]
l = \
[ gmsh.model.occ.addLine(i, j) for i, j in
  [ [p[0], p[1]]
  , [p[1], p[2]]
  , [p[2], p[0]] ] ]
c = gmsh.model.occ.addCurveLoop([l[0], l[1], l[2]])
s = gmsh.model.occ.addPlaneSurface([c])

gmsh.model.occ.synchronize()

gmsh.model.mesh.generate(2)

gmsh.write("p08.geo_unrolled")
gmsh.write("p08.brep")
gmsh.write("p08.msh2")

# e for all elements, find the maximum edge length.
# - prints all elements' tags.
# - for all elements, prints the vertices
# - calculate the edge length
# - generate the model data and write it to a file
# find the leftmost element
# - for all elements, find min(x), and min(y)
# - find the tag with the minimum min(y)
# - multiply 

gmsh.fltk.run()

gmsh.finalize()
