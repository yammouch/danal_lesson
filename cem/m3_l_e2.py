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
    ( [ [0, 3, 5]
      , [0, 4, 5]
      , [1, 2, 5]
      , [1, 4, 5] ] )
    isrc = np.array \
    ( [ [0, 1, 3]
      , [1, 2, 3] ])
    lin = np.array( [ [0, 1] ] )
    v2e, bwh = p04.edge_num_banded(tet)
   #print(v2e)
    pgroups = [('e2', ([0,0,1/w],), isrc), ('b', (), tri), ('v', (), tet)]
    freq = 50
    sol = p04.solve_geom(freq, vrt, pgroups, v2e.nnz, v2e, bwh)
    print(sol)
    print(p04.isrc_v(sol, vrt, lin, v2e, [0,0,1]))
    print(p04.u0*0.5*w*l/h*2*np.pi*freq)

if __name__ == '__main__':
    main()