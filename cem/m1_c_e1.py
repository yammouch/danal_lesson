import numpy as np
import p04

def main():
    np.set_printoptions(precision=3)
    w = 14
    l = 7
    h = 2
    vrt = np.array \
    ( [ [    0,    0, -w/2, w/2 ]
      , [    0,    0,    l,   l ]
      , [ -h/2,  h/2,    0,   0 ] ] )
    tet = np.array \
    ( [ [ 0, 1, 2, 3 ] ] )
    lin = np.array( [ [0, 1] ] )
    v2e, bwh = p04.edge_num_banded(tet)
    pgroups = [('e', ([0,0,1],), tet), ('v', (), tet)]
    freq = 50
    sol = p04.solve_geom(freq, vrt, pgroups, v2e.nnz, v2e, bwh)
    print(sol)
    print(p04.isrc_v(sol, vrt, lin, v2e, [0,0,1]))
    print(2*(1/freq)/4*(2/np.pi)/(4*np.pi*p04.e0*0.5*h))

if __name__ == '__main__':
    main()