import numpy as np
import gmsh
import dolfinx
import mpi4py
import pyvista

class Mesh(object):

    def __init__(self, tgl_xyz=None, tgl_len=None, save=False, show=False):
        super().__init__()
        gmsh.initialize()

        p = \
        [ gmsh.model.occ.addPoint(x, y, 0) for x, y in
          [ [0, 0], [1, 0], [1, 1], [0, 1] ] ]
        l = \
        [ gmsh.model.occ.addLine(i, j) for i, j in
          [ [p[0], p[1]]
          , [p[1], p[2]]
          , [p[2], p[3]]
          , [p[3], p[0]] ] ]
        c = gmsh.model.occ.addCurveLoop([l[0], l[1], l[2], l[3]])
        s = gmsh.model.occ.addPlaneSurface([c])

        gmsh.model.occ.synchronize()

        self.isrc   = gmsh.model.addPhysicalGroup(1, [l[3]])
        self.pec    = gmsh.model.addPhysicalGroup(1, [l[0], l[1], l[2]])
        self.vacuum = gmsh.model.addPhysicalGroup(2, [s])

        gmsh.option.setNumber("Mesh.Algorithm", 8)

        if type(tgl_xyz) == type(None) or type(tgl_len) == type(None):
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 3)
        else:
            td_view = gmsh.view.add("mesh size")
            gmsh.view.addListData \
            ( td_view, "ST", tgl_xyz.shape[0]
            , np.hstack
              ( ( np.moveaxis(tgl_xyz, -1, -2).reshape(-1, 9)
                , tgl_len[..., None].repeat(3, axis=-1) )
              ).reshape(-1) )
            bg_field = gmsh.model.mesh.field.add("PostView")
            gmsh.model.mesh.field.setNumber(bg_field, "ViewTag", td_view)
            gmsh.model.mesh.field.setAsBackgroundMesh(bg_field)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        gmsh.model.mesh.generate(2)
        tgl = list(gmsh.model.mesh.getElementsByType(2))
        tgl[1] = tgl[1].reshape(-1, 3)
        nod_raw = gmsh.model.mesh.getNodes()
        nod = np.full((int(nod_raw[0].max()+1), 3), np.nan, dtype=np.float64)
        nod[nod_raw[0]] = nod_raw[1].reshape(-1, 3)
        tgl_len = \
        ( ( ( nod[tgl[1][..., [0, 0, 1]]]
            - nod[tgl[1][..., [1, 2, 2]]] )
          **2 ) 
        .sum(axis=-1).max(axis=-1)
        **0.5 )

        self.tgl = tgl
        self.nod = nod
        self.tgl_len = tgl_len
        self.msh = dolfinx.io.gmshio.model_to_mesh \
        (gmsh.model, mpi4py.MPI.COMM_SELF, 0)

        if save:
            gmsh.write("p10.geo_unrolled")
            gmsh.write("p10.brep")
            gmsh.write("p10.msh2")
            gmsh.view.write(td_view, "p10.pos")

        if show:
            gmsh.fltk.run()

        gmsh.finalize()

def main():
    msh = Mesh()
    plotter = pyvista.Plotter(shape=(2, 3))
    for i in range(6):
        #msh.tgl_len *= 1.1
        msh.tgl_len[np.argmin(msh.nod.sum(axis=-1)[msh.tgl[1]].min(axis=-1))] *= 0.5
        msh = Mesh(msh.nod[msh.tgl[1]], msh.tgl_len, i==3, False)
        mshv = dolfinx.plot.create_vtk_mesh(msh.msh[0])
        grid = pyvista.UnstructuredGrid(*mshv)
        mark_name = "Marker{}".format(i)
        grid.cell_data[mark_name] = msh.msh[1].values
        grid.set_active_scalars(mark_name)
        plotter.subplot(i//3, i%3)
        plotter.add_text("r{}_c{}".format(i//3, i%3))
        plotter.add_mesh(grid, show_edges=True)
        plotter.view_xy()

    plotter.show()

if __name__ == "__main__":
    main()
