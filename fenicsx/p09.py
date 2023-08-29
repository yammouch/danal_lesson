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

l2 = dolfinx.fem.FunctionSpace(msh, ('DG', 1))
print(l2.dofmap.dof_layout.entity_dofs(0, 2))
print(l2.dofmap.dof_layout.entity_dofs(2, 0))
print(l2.dofmap.dof_layout.entity_closure_dofs(0, 2))
print(l2.dofmap.dof_layout.entity_closure_dofs(2, 0))

l2_t = ufl.TestFunction(l2)

n = ufl.FacetNormal(msh)

l2_npx_v = dolfinx.fem.assemble_vector \
( dolfinx.fem.form
  ( n("+")[0]*l2_t("+")*ufl.dS ) )
print(l2_npx_v.array)
l2_npx = dolfinx.fem.Function(l2)
l2_npx.x.array[:] = l2_npx_v.array

subplotter = pyvista.Plotter(shape=(2, 3))

vmsh = [None] * 6
grid = [None] * 6

vmsh[0] = dolfinx.plot.create_vtk_mesh(l2)
print(vmsh[0])
grid[0] = pyvista.UnstructuredGrid(*vmsh[0])
grid[0].point_data["l2_npx"] = l2_npx.x.array
grid[0].set_active_scalars("l2_npx")

subplotter.subplot(0, 0)
subplotter.add_mesh(grid[0], show_edges=True)
subplotter.view_xy()

subplotter.show()
