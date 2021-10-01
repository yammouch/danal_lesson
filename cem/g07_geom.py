import g07 as dut
import gmsh

gmsh.initialize()
g = dut.make_geom()
print(g)
gmsh.model.occ.synchronize()
gmsh.fltk.run()
gmsh.finalize()