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

vmsh = dolfinx.plot.create_vtk_mesh(h1)
#vmsh = dolfinx.plot.create_vtk_mesh(msh)
print(vmsh)

grid = pyvista.UnstructuredGrid(*vmsh)
subplotter = pyvista.Plotter(shape=(1, 1))
subplotter.subplot(0, 0)
subplotter.add_mesh(grid, show_edges=True)
subplotter.show()
