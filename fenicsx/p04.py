import itertools
import dolfinx
import ufl
import mpi4py
import numpy as np
import pyvista

msh = dolfinx.mesh.create_mesh \
( mpi4py.MPI.COMM_SELF
, [[0,1,3],[0,2,3]]
, [[0,0],[1,0],[0,2],[1,2]]
, ufl.Mesh(ufl.Cell("triangle", 2) ) )

print(msh.geometry.x)

for i, j in itertools.product(range(3), range(3)):
    msh.topology.create_connectivity(i, j)

for i in [1, 2]:
    print(msh.topology.connectivity(i, 0))

h1 = dolfinx.fem.FunctionSpace(msh, ('CG', 1))
print(h1.dofmap.dof_layout.entity_dofs(0, 2))
print(h1.dofmap.dof_layout.entity_dofs(2, 0))
print(h1.dofmap.dof_layout.entity_closure_dofs(0, 2))
print(h1.dofmap.dof_layout.entity_closure_dofs(2, 0))

h1_f = dolfinx.fem.Function(h1)
print(h1_f.x.array)

h1_f.x.array[h1.dofmap.dof_layout.entity_dofs(0, 2)] = [1]
print(h1_f.x.array)

l2 = dolfinx.fem.FunctionSpace(msh, ('DG', 1))
print(l2.dofmap.dof_layout.entity_dofs(0, 2))
print(l2.dofmap.dof_layout.entity_dofs(2, 0))
print(l2.dofmap.dof_layout.entity_closure_dofs(0, 2))
print(l2.dofmap.dof_layout.entity_closure_dofs(2, 0))

l2_f = dolfinx.fem.Function(l2)
print(l2_f.x.array)

l2_f.interpolate(h1_f)
print(l2_f.x.array)

l2_dx = dolfinx.fem.Function(l2)
l2_dx.interpolate \
( dolfinx.fem.Expression
  ( l2_f.dx(0)
  , l2.element.interpolation_points() ) )
print(l2_dx.x.array)

l2_dy = dolfinx.fem.Function(l2)
l2_dy.interpolate \
( dolfinx.fem.Expression
  ( l2_f.dx(1)
  , l2.element.interpolation_points() ) )
print(l2_dy.x.array)

subplotter = pyvista.Plotter(shape=(1, 4))

vmsh = [None] * 4
grid = [None] * 4

vmsh[0] = dolfinx.plot.create_vtk_mesh(h1)
print(vmsh[0])
grid[0] = pyvista.UnstructuredGrid(*vmsh[0])
grid[0].point_data["h1_f"] = h1_f.x.array
grid[0].set_active_scalars("h1_f")
subplotter.subplot(0, 0)
subplotter.add_mesh(grid[0], show_edges=True)
subplotter.view_xy()

vmsh[1] = dolfinx.plot.create_vtk_mesh(l2)
print(vmsh[1])
grid[1] = pyvista.UnstructuredGrid(*vmsh[1])
grid[1].point_data["l2_f"] = l2_f.x.array
grid[1].set_active_scalars("l2_f")
subplotter.subplot(0, 1)
subplotter.add_mesh(grid[1], show_edges=True)
subplotter.view_xy()

vmsh[2] = dolfinx.plot.create_vtk_mesh(l2)
print(vmsh[2])
grid[2] = pyvista.UnstructuredGrid(*vmsh[2])
grid[2].point_data["l2_dx"] = l2_dx.x.array
grid[2].set_active_scalars("l2_dx")
subplotter.subplot(0, 2)
subplotter.add_mesh(grid[2], show_edges=True)
subplotter.view_xy()

vmsh[3] = dolfinx.plot.create_vtk_mesh(l2)
print(vmsh[3])
grid[3] = pyvista.UnstructuredGrid(*vmsh[3])
grid[3].point_data["l2_dy"] = l2_dy.x.array
grid[3].set_active_scalars("l2_dy")
subplotter.subplot(0, 3)
subplotter.add_mesh(grid[3], show_edges=True)
subplotter.view_xy()

subplotter.show()
