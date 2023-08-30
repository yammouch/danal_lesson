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
l2_n = np.empty((2, 2, 2), dtype=np.object_)

for i in itertools.product(*[range(i) for i in [2, 2, 2]]):
    l2_n_v = dolfinx.fem.assemble_vector \
    ( dolfinx.fem.form
      #( l2_t("+-"[i[0]])*ufl.dS ) )
      ( n("+-"[i[2]])[i[1]]*l2_t("+-"[i[0]])*ufl.dS ) )
    print(l2_n_v.array)
    l2_n[i] = dolfinx.fem.Function(l2)
    l2_n[i].x.array[:] = l2_n_v.array

subplotter = pyvista.Plotter(shape=(2, 4))

for i in itertools.product(*[range(i) for i in [2, 2, 2]]):
    vmsh = dolfinx.plot.create_vtk_mesh(l2)
    print(vmsh)
    grid = pyvista.UnstructuredGrid(*vmsh)
    fname = "l2_n{}{}{}".format("+-"[i[2]], "xy"[i[1]], "+-"[i[0]])
    grid.point_data[fname] = l2_n[i].x.array
    grid.set_active_scalars(fname)

    subplotter.subplot(i[2], i[0]+2*i[1])
    subplotter.add_mesh(grid, clim=[-1.1, 1.1], cmap="bwr", show_edges=True)
    subplotter.view_xy()

subplotter.show()
