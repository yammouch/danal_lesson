import numpy as np
import scipy.constants
import gmsh
import dolfinx
import ufl
import mpi4py
import basix
import pyvista
import petsc4py
import p06
import p10

def solve_field(mesh, facet_tags, pec, isrc, w):
    hc = dolfinx.fem.FunctionSpace(mesh, ('N1curl', 1))

    dirichlet_f = dolfinx.fem.Function(hc)
    dirichlet_f.x.array[:] = 0

    dirichlet_bc = dolfinx.fem.dirichletbc \
    ( dirichlet_f
    , ( np.sort
        ( np.hstack
          ( [ dolfinx.fem.locate_dofs_topological(hc, 1, [i])
              for i in facet_tags.find(pec) ] ) ) ) )

    ds = ufl.Measure('ds', mesh, subdomain_data=facet_tags)
    hc_tr = ufl.TrialFunction(hc)
    hc_ts = ufl.TestFunction(hc)
    n = ufl.FacetNormal(mesh)
    a = ufl.inner(ufl.curl(hc_tr), ufl.curl(hc_ts))/scipy.constants.mu_0*ufl.dx
    a -= w**2*scipy.constants.epsilon_0*ufl.inner(hc_tr, hc_ts)*ufl.dx
    L = ufl.inner(0*n[1]-1*n[0], hc_ts[0]*n[1]-hc_ts[1]*n[0])*-w*ds(isrc)
    problem = dolfinx.fem.petsc.LinearProblem(a, L, [dirichlet_bc])
    sol = problem.solve()
    return sol


def main():
    mshg = p10.Mesh()

    print(petsc4py.PETSc.ScalarType)

    w = 2*scipy.constants.pi*1
    sol = solve_field(mshg.msh[0], mshg.msh[2], mshg.pec, mshg.isrc, w)
    print(sol.x.array)

    l2_2d = dolfinx.fem.VectorFunctionSpace(mshg.msh[0], ("DG", 2))
    l2_2d_f = dolfinx.fem.Function(l2_2d)
    l2_2d_f.interpolate(sol)

    n = ufl.FacetNormal(mshg.msh[0])
    ds = ufl.Measure('ds', mshg.msh[0], subdomain_data=mshg.msh[2])
    er = dolfinx.fem.FunctionSpace \
    ( mshg.msh[0]
    , basix.ufl_wrapper.BasixElement(p06.element) )
    er_tr = ufl.TrialFunction(er)
    er_ts = ufl.TestFunction(er)

    a_er = ufl.inner(ufl.curl(er_tr), ufl.curl(er_ts))/scipy.constants.mu_0*ufl.dx
    a_er -= w**2*scipy.constants.epsilon_0*ufl.inner(er_tr, er_ts)*ufl.dx
    L_er = -ufl.inner(ufl.curl(sol), ufl.curl(er_ts))/scipy.constants.mu_0*ufl.dx
    L_er += w**2*scipy.constants.epsilon_0*ufl.inner(sol, er_ts)*ufl.dx
    L_er -= ufl.inner(0*n[1]-1*n[0], er_ts[0]*n[1]-er_ts[1]*n[0])*-w*ds(mshg.isrc)
    L_er -= ufl.inner( n[1]*ufl.curl(sol), er_ts[0])/scipy.constants.mu_0*ds(mshg.pec)
    L_er -= ufl.inner(-n[0]*ufl.curl(sol), er_ts[1])/scipy.constants.mu_0*ds(mshg.pec)
    L_er -= ufl.inner \
    (   n("+")[1]
      * 0.5
      * (ufl.curl(sol)("+") + ufl.curl(sol)("-"))
    , er_ts("+")[0] )/scipy.constants.mu_0*ufl.dS
    L_er -= ufl.inner \
    (   n("-")[1]
      * 0.5
      * (ufl.curl(sol)("+") + ufl.curl(sol)("-"))
    , er_ts("-")[0] )/scipy.constants.mu_0*ufl.dS
    problem_er = dolfinx.fem.petsc.LinearProblem(a_er, L_er, [])
    sol_er = problem_er.solve()
    l2_2d_er = dolfinx.fem.Function(l2_2d)
    l2_2d_er.interpolate(sol_er)
    print(sol_er.x.array)

    cells, cell_types, x = dolfinx.plot.create_vtk_mesh(l2_2d)

    plotter = pyvista.Plotter(shape=(1, 2))

    grid = pyvista.UnstructuredGrid(cells, cell_types, x)
    grid.point_data["l2_2d_f"] = np.hstack \
    ( [ l2_2d_f.x.array.reshape \
        ( x.shape[0], l2_2d.dofmap.index_map_bs )
      , np.zeros((x.shape[0], 1)) ] )
    glyphs = grid.glyph(orient="l2_2d_f", factor=1e5)

    plotter.subplot(0, 0)
    plotter.add_mesh \
    ( pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(mshg.msh[0]))
    , style="wireframe", line_width=2, color="black" )
    plotter.add_mesh(glyphs)
    plotter.view_xy()

    grid1 = pyvista.UnstructuredGrid(cells, cell_types, x)
    grid1.point_data["l2_2d_er"] = np.hstack \
    ( [ l2_2d_er.x.array.reshape \
        ( x.shape[0], l2_2d.dofmap.index_map_bs )
      , np.zeros((x.shape[0], 1)) ] )
    glyphs1 = grid1.glyph(orient="l2_2d_er", factor=1e5)

    plotter.subplot(0, 1)
    plotter.add_mesh \
    ( pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(mshg.msh[0]))
    , style="wireframe", line_width=2, color="black" )
    plotter.add_mesh(glyphs1)
    plotter.view_xy()

    plotter.show()

if __name__ == "__main__":
    main()
