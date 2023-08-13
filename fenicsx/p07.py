import numpy as np
import dolfinx
import ufl
import mpi4py
import petsc4py

print(petsc4py.PETSc.ScalarType)

msh = dolfinx.mesh.create_mesh \
( mpi4py.MPI.COMM_WORLD
, ((0, 1, 2),)
, ((0, -1), (0, 1), (2, 0))
, ufl.Mesh(ufl.Cell('triangle')) )

hc = dolfinx.fem.FunctionSpace(msh, ('N1curl', 1))

neumann = dolfinx.mesh.locate_entities \
( msh, 1, lambda x: np.isclose(x[0], 0.0) )

mt = dolfinx.cpp.mesh.MeshTags_int32 \
( msh, 1, neumann, np.full_like(neumann, 0) )

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
a = ufl.inner(ufl.curl(hc_tr), ufl.curl(hc_ts))*ufl.dx
L = ufl.inner(0*n[1]-1*n[0], hc_ts[0]*n[1]-hc_ts[1]*n[0])*ds(0)
problem = dolfinx.fem.petsc.LinearProblem(a, L, [dirichlet_bc])
sol = problem.solve()
print(sol.x.array)
