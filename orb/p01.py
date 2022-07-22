import numpy as np

def kin(k_basis, lc):
    k = 2*np.pi/np.array(lc)[None, :]*k_basis
    x = k[:, None, :] - k[None, :, :]
    x = x**2
    x = x.sum(axis=-1)
    x = -0.5*x
    return x

def symmetrize(a):
    for i in range(a.ndim):
        l = a.shape[i]
        if l % 2 == 0:
            a[(slice(None),)*i + (l//2,)] *= 0.5
    return a

def cou_idx(k_basis):
    kdic = dict \
    ( zip
      ( (tuple(x) for x in k_basis)
      , range(len(k_basis)) ) )
    retval = []
    for p in k_basis:
        row = []
        for q in k_basis:
            r = k_basis
            t = q - p
            if tuple(t) not in kdic:
                continue
            s = r - t[None, :]
            t = tuple(t)
            rn, sn, tn = [], [], []
            for r1, s1 in zip(r, s):
                r1 = tuple(r1); s1 = tuple(s1)
                if r1 not in k_basis:
                    continue
                if s1 not in k_basis:
                    continue
                rn.append(kdic[r1])
                sn.append(kdic[s1])
                tn.append(kdic[t])
            row.append([np.array(x) for x in [rn, sn, tn]])
        retval.append(row)
    return retval
