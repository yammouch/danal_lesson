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
    lacc = [(p01.volume(0, p04.e0, p04.u0), tet)]
    racc = [(p01.isrc(1, [0,0,1]), tet)]
    pec  = [(lambda f, p: None, tri)]
    solver = p04.Banded(vrt, lacc, racc, pec)
    freq = 50
    sol = solver.solve(freq)
    print(sol)
    print(p04.isrc_v(sol, vrt, lin, solver.v2e, [0,0,1]))
    print(p04.u0*0.5*w*l/h*2*np.pi*freq)

if __name__ == '__main__':
    main()