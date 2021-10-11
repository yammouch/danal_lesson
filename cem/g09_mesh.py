import gmsh
import g09 as dut

def pp_nodes(n):
  print(n[0])
  print(n[1].reshape(-1, 3))
  print(n[2])

def pp_elem(e):
  print('physical group: ', e[0])
  print('type: ', e[1])
  print('tags:')
  for x in e[2]:
    print(x)
  print('nodes:')
  for x in e[3]:
    if e[1][0] == 4:
      print(x.reshape(-1, 4))
    elif e[1][0] == 2:
      print(x.reshape(-1, 3))
    else:
      print(x)

gmsh.initialize()

air_tag, cond_tag, isrc_tag, probe_tag = dut.make_geom()
print(air_tag, cond_tag, isrc_tag, probe_tag)

gmsh.model.occ.synchronize()
isrc, cond, air, probe = dut.assign_physicals(air_tag, cond_tag, isrc_tag, probe_tag)
print(isrc, cond, air, probe)

#gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 3)
#gmsh.option.setNumber("Mesh.Algorithm", 8)
nodes, elems = dut.gen_mesh()
print('-' * 40)
pp_nodes(nodes)
print('-' * 40)
for e in elems:
  pp_elem(e)

gmsh.fltk.run()
gmsh.finalize()