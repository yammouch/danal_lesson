import numpy as np
import p02

def orb_1d():
    basis_nwn = np.mgrid[-4:5].reshape(-1, 1)
    box_size = np.full(1, 2*np.pi)
    vext = np.full((16,), 20, dtype=float)
    vext[0:7] = 0
    e, v = p02.solve_1elec(basis_nwn, box_size, vext)
    print(e)

def orb_2d():
    basis_nwn = np.mgrid[-4:5,-4:5].reshape(2, -1).T
    box_size = np.full(2, 2*np.pi)
    vext = np.full((16, 16), 20, dtype=float)
    vext[0:7, 0:7] = 0
    e, v = p02.solve_1elec(basis_nwn, box_size, vext)
    print(e)

def orb_3d():
    basis_nwn = np.mgrid[-4:5,-4:5,-4:5].reshape(3, -1).T
    box_size = np.full(3, 2*np.pi)
    vext = np.full((16, 16, 16), 20, dtype=float)
    vext[0:7, 0:7, 0:7] = 0
    e, v = p02.solve_1elec(basis_nwn, box_size, vext)
    print(e)

def hydrogen_vext(npoint_half, size):
    side = 0.5*size*(1-1/(2*npoint_half))
    dist = np.linspace(-side, side, 2*npoint_half)
    d2 = dist**2
    d2_3d = d2[:, None, None] + d2[None, :, None] + d2[None, None, :]
    vext = d2_3d**0.5
    vext = -1.0/vext
    return vext

def hydrogen():
    #npoint_half = 8
    #npoint_half = 12
    npoint_half = 16
    size = 8
    sl = slice(-npoint_half//2, npoint_half//2 + 1)
    basis_nwn = np.mgrid[sl, sl, sl].reshape(3, -1).T
    box_size = np.full(3, size, dtype=float)
    vext = hydrogen_vext(npoint_half, size)
    e, v = p02.solve_1elec(basis_nwn, box_size, vext)
    print(e)

def main():
    orb_1d()
    orb_2d()
    orb_3d()
    hydrogen()

if __name__ == '__main__':
    main()
