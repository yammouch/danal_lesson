import numpy as np
import p04

def main():
    np.set_printoptions(precision=3)
    vrt = np.array \
    ( [ [ 0, 1, 0, 0, 1, 0 ]
      , [ 0, 0, 1, 0, 0, 1 ]
      , [ 0, 0, 0, 1, 1, 1 ] ] )
    tet = np.array \
    ( [ [ 0, 1, 2, 3 ]
      , [ 1, 2, 3, 4 ]
      , [ 2, 3, 4, 5 ] ] )
    tri = np.array \
    ( [ [0, 1, 2]
      , [3, 4, 5] ] )
    lin = np.array( [ [0, 3] ] )
    v2e, bwh = p04.edge_num_banded(tet)
    pgroups = [('e', ([0,0,1],), lin), ('b', (), tri), ('v', (), tet)]
    for freq in [10e3, 100e3, 1e6]:
        sol = p04.solve_geom(freq, vrt, pgroups, v2e.nnz, v2e, bwh)
        print(sol)
        for ptype, attr, nodes in pgroups:
            if ptype == 'e':
                print(p04.isrc_v(sol, vrt, nodes, v2e, attr[0]))
        print(1/(p04.e0*0.5*2*np.pi*freq))

if __name__ == '__main__':
    main()