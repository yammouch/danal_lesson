import numpy as np
import p04 as dut

tet = np.array \
( [ [0, 1, 2, 4]
  , [1, 2, 3, 4]
  , [0, 1, 4, 5] ] )

dut.edge_num_banded(tet)