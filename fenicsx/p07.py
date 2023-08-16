import numpy as np
import dolfinx
import ufl
import mpi4py
import petsc4py
import pyvista
import scipy.constants
import p06
import basix

print(petsc4py.PETSc.ScalarType)

msh = dolfinx.mesh.create_mesh \
( mpi4py.MPI.COMM_WORLD
, ((0, 1, 2),)
, ((0, -1), (0, 1), (2, 0))
, ufl.Mesh(ufl.Cell('triangle')) )

hc = dolfinx.fem.FunctionSpace(msh, ('N1curl', 1))

neumann = dolfinx.mesh.locate_entities \
( msh, 1, lambda x: np.isclose(x[0], 0.0) )

tags = np.zeros(3, dtype=np.int32)
tags[np.isin(np.arange(3), neumann)] = 1

mt = dolfinx.cpp.mesh.MeshTags_int32 \
( msh, 1, np.arange(3, dtype=np.int32), tags )
#( msh, 1, neumann, np.full_like(neumann, 0) )

dirichlet_f = dolfinx.fem.Function(hc)
dirichlet_f.x.array[:] = 0

dirichlet_bc = dolfinx.fem.dirichletbc \
( dirichlet_f
, ( np.sort
    ( np.hstack
      ( [ dolfinx.fem.locate_dofs_topological(hc, 1, [i])
          for i in np.setdiff1d(np.arange(3), neumann) ] ) ) ) )

ds = ufl.Measure('ds', msh, subdomain_data=mt)
hc_tr = ufl.TrialFunction(hc)
hc_ts = ufl.TestFunction(hc)
n = ufl.FacetNormal(msh)
w = 2*scipy.constants.pi*1
a = ufl.inner(ufl.curl(hc_tr), ufl.curl(hc_ts))/scipy.constants.mu_0*ufl.dx
a -= w**2*scipy.constants.epsilon_0*ufl.inner(hc_tr, hc_ts)*ufl.dx
L = ufl.inner(0*n[1]-1*n[0], hc_ts[0]*n[1]-hc_ts[1]*n[0])*-w*ds(1)
problem = dolfinx.fem.petsc.LinearProblem(a, L, [dirichlet_bc])
sol = problem.solve()
print(sol.x.array)

l2_2d = dolfinx.fem.VectorFunctionSpace(msh, ("DG", 2))
l2_2d_f = dolfinx.fem.Function(l2_2d)
l2_2d_f.interpolate(sol)

cells, cell_types, x = dolfinx.plot.create_vtk_mesh(l2_2d)
grid = pyvista.UnstructuredGrid(cells, cell_types, x)
grid.point_data["l2_2d_f"] = np.hstack \
( [ l2_2d_f.x.array.reshape \
    ( x.shape[0], l2_2d.dofmap.index_map_bs )
  , np.zeros((x.shape[0], 1)) ] )
glyphs = grid.glyph(orient="l2_2d_f", factor=1e5)

plotter = pyvista.Plotter()
plotter.add_mesh \
( pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(msh))
, style="wireframe", line_width=2, color="black" )
plotter.add_mesh(glyphs)
plotter.view_xy()
plotter.show()

er = dolfinx.fem.FunctionSpace \
( msh
, basix.ufl_wrapper.BasixElement(p06.element) )
er_tr = ufl.TrialFunction(er)
er_ts = ufl.TestFunction(er)

a_er = ufl.inner(ufl.curl(er_tr), ufl.curl(er_ts))/scipy.constants.mu_0*ufl.dx
a_er -= w**2*scipy.constants.epsilon_0*ufl.inner(er_tr, er_ts)*ufl.dx
L_er = -ufl.inner(ufl.curl(sol), ufl.curl(er_ts))/scipy.constants.mu_0*ufl.dx
L_er += w**2*scipy.constants.epsilon_0*ufl.inner(sol, er_ts)*ufl.dx
L_er -= ufl.inner(0*n[1]-1*n[0], er_ts[0]*n[1]-er_ts[1]*n[0])*-w*ds(1)
L_er -= ufl.inner( n[1]*ufl.curl(sol), er_ts[0])/scipy.constants.mu_0*ds(0)
L_er -= ufl.inner(-n[0]*ufl.curl(sol), er_ts[1])/scipy.constants.mu_0*ds(0)
problem_er = dolfinx.fem.petsc.LinearProblem(a_er, L_er, [])
sol_er = problem_er.solve()
print(sol_er.x.array)
