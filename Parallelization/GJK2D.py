import numpy as np
import matplotlib.pyplot as plt
from animate import Target_body, Chaser_body
from numba import njit, prange


@njit(cache=True)
def collison_check(time_array, Target_state_array, Chaser_state_array,
                   dummy=0):

    if dummy == 1:
        return 0.0

    for i in range(time_array.size):
        t = time_array[i]
        Target_pos = Target_state_array[i, 0:2]
        Target_theta = Target_state_array[i, 4]
        Chaser_pos = Chaser_state_array[i, 0:2]
        Chaser_theta = Chaser_state_array[i, 4]

        Tx_coord, Ty_coord = Target_body(Target_pos, Target_theta)
        Cx_coord, Cy_coord = Chaser_body(Chaser_pos, Chaser_theta)

        Target_shape = np.vstack((Tx_coord, Ty_coord)).T
        Chaser_shape = np.vstack((Cx_coord, Cy_coord)).T

        out = isCollide(Target_shape, Chaser_shape)
        if out == 1:
            print('Collision detected at ', t, ' s')
            return t

    if i == time_array.size-1:
        print('No Collision detected')
        return time_array[-1]


# a and b are shapes' coordinates

@njit
def minkowskiDiff(a, b):
    a_len = a.shape[0]
    b_len = b.shape[0]

    mDiff = np.zeros((a_len*b_len, 2))
    k = 0
    for i in range(a_len):
        for j in range(b_len):
            mDiff[k, 0] = a[i, 0] - b[j, 0]
            mDiff[k, 1] = a[i, 1] - b[j, 1]
            k += 1
    return mDiff


@njit
def convexHull(a, b):
    mDiff = minkowskiDiff(a, b)

    hull_ind = -1*np.ones(mDiff.shape[0], dtype='int')
    k = 0
    for deg in range(360):
        rad = (deg) * np.pi/180
        directionVector = np.array([np.cos(rad), np.sin(rad)])

        indMax = np.argmax(np.dot(mDiff, directionVector))

        if indMax not in hull_ind:
            hull_ind[k] = indMax
            k += 1

    hull = mDiff[hull_ind[:k]]

    return mDiff, hull


@njit
def subtended_angle(c_hull, point):

    hull = np.zeros((c_hull.shape[0] + 1, c_hull.shape[1]))
    hull[0:c_hull.shape[0], :] = c_hull
    hull[c_hull.shape[0], :] = c_hull[0, :]

    Nvert = hull.shape[0]
    Nedge = Nvert - 1

    Totang = 0
    for i in range(Nedge):
        v1 = point - hull[i, :]
        v2 = point - hull[i+1, :]
        ang = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        if ang > 1.0:
            ang = 1.0
        ang = np.rad2deg(np.arccos(ang))
        Totang += ang

    return Totang


@njit
def isCollide(a, b):

    mdiff, hull = convexHull(a, b)

    normvec = np.zeros(hull.shape[0])
    for i in range(normvec.size):
        normvec[i] = np.linalg.norm(hull)

    if normvec[normvec < 1e-3].size > 0:
        return 1
    elif np.abs(subtended_angle(hull, np.zeros(2)) - 360.0) < 1e-3:
        return 1
    else:
        return 0


def pltArray(arr):
    x = []
    y = []
    for pt in arr:
        x.append(pt[0])
        y.append(pt[1])
    x.append(x[0])
    y.append(y[0])
    plt.plot(x, y, 'g-')
    plt.plot(x, y, 'go')


def sctrArray(arr):
    x = []
    y = []
    for pt in arr:
        x.append(pt[0])
        y.append(pt[1])
    x.append(x[0])
    y.append(y[0])
    plt.scatter(x, y)


## Old Code

# sq1 = [[0,0], [0,1], [1,1], [1,0], [0.5, -0.5]]
# sq2 = [[2,2], [2,3], [3,3], [3,2]]
# sq3 = [[1,1], [2,1], [2,2], [1,2]]
# tr1 = [[0, 0.5], [-1, -2], [-1,2]]

# mDiff, hull = convexHull(sq1, tr1)

# print(isCollide(sq1, tr1))

# pltArray(hull)
# pltArray(sq1)
# pltArray(tr1)

# plt.show()
