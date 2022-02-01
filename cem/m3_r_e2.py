import numpy as np
import p01
import p04

def main():
    np.set_printoptions(precision=3)
    w = 1
    l = 1
    h = 1
    vrt = np.array \
    ( [ [-w/2, 0, -h/2]
      , [-w/2, 0,  h/2]
      , [ w/2, 0,  h/2]
      , [ w/2, 0, -h/2]
      , [-w/2, l,  0  ]
      , [ w/2, l,  0  ] ] )
    tet = np.array \
    ( [ [0, 1, 3, 5]
      , [1, 2, 3, 5]
      , [0, 1, 4, 5] ] )
    tri = np.array \
    ( [ [0, 1, 4]
      , [2, 3, 5] ] )
    isrc = np.array \
    ( [ [0, 1, 3]
      , [1, 2, 3] ])
    lin = np.array( [ [0, 3] ] )
    lacc = [(p01.volume(1/140e-8, p04.e0, p04.u0), tet)]
    racc = [(p01.isrc(2, [1/h,0,0]), isrc)]
    pec  = [(lambda f, p: None, tri) ]
    solver = p04.Banded(vrt, lacc, racc, pec)
    freq = 50
    sol = solver.solve(freq)
    print(sol)
    print(p04.isrc_v(sol, vrt, lin, solver.v2e, [1,0,0]))
    print(140e-8*w/(0.5*h*l))

if __name__ == '__main__':
    main()