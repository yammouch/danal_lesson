import dolfinx
import ufl
import mpi4py
import numpy as np
import itertools

msh = dolfinx.mesh.create_mesh \
( mpi4py.MPI.COMM_SELF
, [[0,1,2],[0,3,2]]
, [[0,0],[1,0],[0,1],[1,1]]
, ufl.Mesh(ufl.Cell("triangle", 2) ) )

print(msh.geometry.x)

for i, j in itertools.product(range(3), range(3)):
    msh.topology.create_connectivity(i, j)

for i in [1, 2]:
    print(msh.topology.connectivity(i, 0))

