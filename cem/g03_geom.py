import g03 as dut
import gmsh

gmsh.initialize()
dut.make_geom()
gmsh.model.occ.synchronize()
gmsh.fltk.run()
gmsh.finalize()