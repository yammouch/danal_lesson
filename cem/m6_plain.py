import numpy as np
import p01
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
    lacc = [ (p01.absorb, isrc)
           , (p01.absorb, absorb)
           , (p01.volume(0, p04.e0, p04.u0), tet) ]
    racc = [ (p01.isrc(2, [0,0,1]), isrc) ]
    pec  = [ (lambda f, p: None, pec) ]
    for s in [p04.Square, p04.Banded]:
      solver = s(vrt, lacc, racc, pec)
      print(solver.v2e)
      freq = 1e6
      sol = solver.solve(freq)
      print(sol)
      print(0.5*(p04.u0/p04.e0)**0.5*h)
      print(-2*np.pi*freq*l/p04.c*180/np.pi)
      v_src = p04.isrc_v(sol, vrt, np.array([[0, 1]]), solver.v2e, [0,0,1])
      print(v_src)
      print(np.angle(-v_src)*180/np.pi)
      v_out = p04.isrc_v(sol, vrt, np.array([[4, 5]]), solver.v2e, [0,0,1])
      print(v_out)
      print(np.abs(v_out))
      print(np.angle(-v_out)*180/np.pi)

if __name__ == '__main__':
    main()