import numpy as np
import p01
import p04

def main():
    np.set_printoptions(precision=3)
    w = 4
    l = 14
    h = 2
    vrt = np.array \
    ( [ [   0, 0, -h/2]
      , [   0, 0,  h/2]
      , [-w/2, l,    0]
      , [ w/2, l,    0] ] )
    tet = np.array \
    ( [ [0, 1, 2, 3] ] )
    tri = np.array \
    ( [ [0, 2, 3]
      , [1, 2, 3] ] )
    lin = np.array( [ [0, 1] ] )
    v2e, bwh = p04.edge_num_banded(tet)
    pgroups = [ (p04.racc, p01.isrc(1, [0,0,1]), tet)
              , (p04.lacc, p01.volume(0, p04.e0, p04.u0), tet)
              , (p04.pec, lambda f, p: None, tri)]
    freq = 50
    sol = p04.solve_geom(freq, vrt, pgroups, v2e.nnz, v2e, bwh)
    print(sol)
    print(p04.isrc_v(sol, vrt, lin, v2e, [0,0,1]))
    print(p04.u0*0.5*w*l/h*2*np.pi*freq)

if __name__ == '__main__':
    main()