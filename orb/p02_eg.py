import numpy as np
import p02

def orb_1d():
    basis_nwn = np.mgrid[-4:5].reshape(-1, 1)
    box_size = np.full(1, 2*np.pi)
    vext = np.full((16,), 20, dtype=float)
    vext[0:7] = 0
    e, v = p02.solve_1elec(basis_nwn, box_size, vext)
    print(e[:20])

def orb_2d():
    basis_nwn = np.mgrid[-4:5,-4:5].reshape(2, -1).T
    box_size = np.full(2, 2*np.pi)
    vext = np.full((16, 16), 20, dtype=float)
    vext[0:7, 0:7] = 0
    e, v = p02.solve_1elec(basis_nwn, box_size, vext)
    print(e[:20])

def orb_3d():
    basis_nwn = np.mgrid[-4:5,-4:5,-4:5].reshape(3, -1).T
    box_size = np.full(3, 2*np.pi)
    vext = np.full((16, 16, 16), 20, dtype=float)
    vext[0:7, 0:7, 0:7] = 0
    e, v = p02.solve_1elec(basis_nwn, box_size, vext)
    print(e[:20])

def main():
    orb_1d()
    orb_2d()
    orb_3d()

if __name__ == '__main__':
    main()
