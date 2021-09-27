import g04 as dut
import gmsh

gmsh.initialize()
g = dut.make_geom(1, 1, 1)
print(g)
gmsh.model.occ.synchronize()
gmsh.fltk.run()
gmsh.finalize()