import numpy as np

def quatmultiply(q, p):
    qs = q[0]
    qv = q[1:4]
    ps = p[0]
    pv = p[1:4]

    ret = np.zeros_like(q, dtype='double')
    ret[0] = ps*qs - np.dot(pv, qv)
    ret[1:4] = qs*pv + ps*qv + np.cross(qv, pv)
    return ret


def quatconj(q):
    ret = np.copy(q)
    ret[1:4] = -1.0*ret[1:4]
    return ret


def quatrotate(q, v):
    vq = np.zeros_like(q, dtype='double')
    vq[1:4] = v
    ret1 = quatmultiply(q, vq)
    ret2 = quatmultiply(ret1, quatconj(q))
    return ret2[1:4]
