import numpy as np
import gmsh

gmsh.initialize()

p = \
[ gmsh.model.occ.addPoint(x, y, 0) for x, y in
  [ [0, 0], [1, 0], [1, 1], [0, 1] ] ]
l = \
[ gmsh.model.occ.addLine(i, j) for i, j in
  [ [p[0], p[1]]
  , [p[1], p[2]]
  , [p[2], p[3]]
  , [p[3], p[0]] ] ]
c = gmsh.model.occ.addCurveLoop([l[0], l[1], l[2], l[3]])
s = gmsh.model.occ.addPlaneSurface([c])
p = gmsh.model.addPhysicalGroup(2, [s])

gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.Algorithm", 8)
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 3)
gmsh.model.mesh.generate(2)
print(gmsh.model.mesh.getNodesByElementType(2))
td_view = gmsh.view.add("test data")
tgl = list(gmsh.model.mesh.getElementsByType(2))
tgl[1] = tgl[1].reshape(-1, 3)
nod_raw = gmsh.model.mesh.getNodes()
nod = np.full((int(nod_raw[0].max()+1), 3), np.nan, dtype=np.float64)
nod[nod_raw[0]] = nod_raw[1].reshape(-1, 3)
tgl_len = \
( ( ( nod[tgl[1][..., [0, 0, 1]]]
    - nod[tgl[1][..., [1, 2, 2]]] )
  **2 ) 
.sum(axis=-1).max(axis=-1)
**0.5 )
print(tgl_len)
gmsh.view.addModelData \
( td_view, 0, "", "ElementData"
, tgl[0], tgl_len[..., None] )

gmsh.write("p08.geo_unrolled")
gmsh.write("p08.brep")
gmsh.write("p08.msh2")
gmsh.view.write(td_view, "p08.pos")

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
