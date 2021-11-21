import numpy as np
import p04

def main():
    np.set_printoptions(precision=3)
    w = 1
    l = 1
    h = 1
    vrt = np.array \
    ( [ [ -w/2, -w/2, w/2,  w/2, -w/2, w/2 ]
      , [    0,    0,   0,    0,    l,   l ]
      , [ -h/2,  h/2, h/2, -h/2,    0,   0 ] ] )
    tet = np.array \
    ( [ [ 0, 1, 3, 5 ]
      , [ 1, 2, 3, 5 ]
      , [ 0, 1, 4, 5 ] ] )
    tri = np.array \
    ( [ [0, 1, 4]
      , [2, 3, 5] ] )
    lin = np.array( [ [0, 3] ] )
    v2e, bwh = p04.edge_num_banded(tet)
    pgroups = [ ('e3', ([1/(0.5*h*l),0,0],), tet)
              , ('v', (1/140e-8, p04.e0, p04.u0), tet)]
    freq = 50
    sol = p04.solve_geom(freq, vrt, pgroups, v2e.nnz, v2e, bwh)
    print(sol)
    print(p04.isrc_v(sol, vrt, lin, v2e, [1,0,0]))
    print(140e-8*w/(0.5*h*l))

if __name__ == '__main__':
    main()