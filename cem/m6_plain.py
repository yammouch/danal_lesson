import numpy as np
import p04

def main():
    #np.set_printoptions(precision=3)
    w = 1
    l = 1
    h = 1
    vrt = np.array \
    ( [ [-w/2, 0, -h/2]
      , [-w/2, 0,  h/2]
      , [ w/2, 0,  h/2]
      , [ w/2, 0, -h/2]
      , [-w/2, l, -h/2]
      , [-w/2, l,  h/2]
      , [ w/2, l,  h/2]
      , [ w/2, l, -h/2] ] )
    tet = np.array \
    ( [ [0, 1, 3, 7]
      , [0, 1, 4, 7]
      , [1, 2, 3, 7]
      , [1, 2, 6, 7]
      , [1, 4, 5, 7]
      , [1, 5, 6, 7] ] )
    pec = np.array \
    ( [ [0, 3, 7]
      , [0, 4, 7]
      , [1, 2, 6]
      , [1, 5, 6] ] )
    isrc = np.array \
    ( [ [0, 1, 3]
      , [1, 2, 3] ])
    absorb = np.array \
    ( [ [4, 5, 7]
      , [5, 6, 7] ])
    v2e, bwh = p04.edge_num_banded(tet)
    print(v2e)
    pgroups = [ ('e2', ([0,0,1],), isrc)
              , ('b', (), pec)
              , ('a', (), isrc)
              , ('a', (), absorb)
              , ('v', (0, p04.e0, p04.u0), tet)]
    freq = 1e6
    sol = p04.solve_geom(freq, vrt, pgroups, v2e.nnz, v2e, bwh)
    print(sol)
    print(0.5*(p04.u0/p04.e0)**0.5*h)
    print(-2*np.pi*freq*l/p04.c*180/np.pi)
    v_src = p04.isrc_v(sol, vrt, np.array([[0, 1]]), v2e, [0,0,1])
    print(v_src)
    print(np.angle(-v_src)*180/np.pi)
    v_out = p04.isrc_v(sol, vrt, np.array([[4, 5]]), v2e, [0,0,1])
    print(v_out)
    print(np.abs(v_out))
    print(np.angle(-v_out)*180/np.pi)

if __name__ == '__main__':
    main()