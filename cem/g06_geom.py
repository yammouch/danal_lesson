import g05 as dut
import gmsh

gmsh.initialize()
g = dut.make_geom(1, 0.01)
print(g)
gmsh.model.occ.synchronize()
gmsh.fltk.run()
gmsh.finalize()