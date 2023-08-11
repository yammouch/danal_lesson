import itertools
import numpy as np
import basix

pts, wts = basix.make_quadrature(basix.CellType.triangle, 4)
poly = basix.tabulate_polynomials \
( basix.PolynomialType.legendre
, basix.CellType.triangle
, 2
, pts )

wcoeffs = np.array \
( [ ((f(pts[:, 0], pts[:, 1])*wts)[None, :]*poly).sum(axis=-1) for f in
    [ lambda x, y: y*(1-y)
    , lambda x, y: x*y
    , lambda x, y: x*y
    , lambda x, y: x*(1-x) ] ]
, dtype=np.float64) \
.reshape(2, -1)

x = [[], [], [], []]
for i, _ in itertools.product(range(2), range(3)):
  x[i].append(np.zeros((0, 2)))
x[2].append(np.array([[0.5, 0.0], [0.0, 0.5]]))

M = [[], [], [], []]
for i, _ in itertools.product(range(2), range(3)):
  M[i].append(np.zeros((0, 2, 0, 1)))
M[2].append \
( np.array
  ( [ [ [ [0.], [0.] ]
      , [ [4.], [0.] ] ]
    , [ [ [0.], [4.] ]
      , [ [0.], [0.] ] ] ] ) )

element = basix.create_custom_element \
( basix.CellType.triangle
, [2], wcoeffs, x, M, 0
, basix.MapType.covariantPiola
, basix.SobolevSpace.HCurl
, True, 1, 2 )

tab = element.tabulate \
( 0
, np.array \
  ( [ [0.0    , 0.0    ]
    , [1.0/3.0, 0.0    ]
    , [1.0/2.0, 0.0    ]
    , [2.0/3.0, 0.0    ]
    , [1.0    , 0.0    ]
    , [0.0    , 0.0    ]
    , [0.0    , 1.0/3.0]
    , [0.0    , 1.0/2.0]
    , [0.0    , 2.0/3.0]
    , [0.0    , 1.0    ]
    , [1.0/3.0, 2.0/3.0]
    , [1.0/2.0, 1.0/2.0]
    , [2.0/3.0, 1.0/3.0] ] ) )
