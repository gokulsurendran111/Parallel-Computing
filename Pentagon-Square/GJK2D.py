import numpy as np
import matplotlib.pyplot as plt
from animate import Target_body, Chaser_body


def collison_check(time_array, Target_state_array, Chaser_state_array):
    for i in range(time_array.size):
        t = time_array[i]
        Target_pos = Target_state_array[i, 0:2]
        Target_theta = Target_state_array[i, 4]
        Chaser_pos = Chaser_state_array[i, 0:2]
        Chaser_theta = Chaser_state_array[i, 4]

        Tx_coord, Ty_coord = Target_body(Target_pos, Target_theta)
        Cx_coord, Cy_coord = Chaser_body(Chaser_pos, Chaser_theta)

        Target_shape = []
        for j in range(Tx_coord.size):
            Target_shape.append([Tx_coord[j], Ty_coord[j]])

        Chaser_shape = []
        for j in range(Cx_coord.size):
            Chaser_shape.append([Cx_coord[j], Cy_coord[j]])

        out = isCollide(Target_shape, Chaser_shape)
        if out is True:
            print('Collision detected at ' + str(t) + 's')
            return t

    if i == time_array.size-1:
        print('No Collision detected')
        return time_array[-1]


# a and b are shapes' coordinates


def minkowskiDiff(a, b):
    a_len = len(a)
    b_len = len(b)

    mDiff = []
    for i in range(a_len):
        for j in range(b_len):
            mDiff.append([a[i][0] - b[j][0], a[i][1] - b[j][1]])

    return mDiff


def convexHull(a, b):
    mDiff = minkowskiDiff(a, b)

    hull = []
    for deg in range(360):
        rad = (deg) * np.pi/180
        directionVector = [np.cos(rad), np.sin(rad)]

        indMax = np.argmax(np.dot(mDiff, directionVector))

        if mDiff[indMax] not in hull:
            hull.append(mDiff[indMax])

    return mDiff, hull


def subtended_angle(hull, point):
    hull.append(hull[0])
    hull = np.array(hull)
    hull_x_coord = hull[:, 0]
    hull_y_coord = hull[:, 1]

    Nvert = hull.shape[0]
    Nedge = Nvert - 1

    Totang = 0
    for i in range(Nedge):
        v1 = point - hull[i]
        v2 = point - hull[i+1]
        ang = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        ang = np.rad2deg(np.arccos(ang))
        Totang += ang

    return Totang


def isCollide(a, b):
    if type(a) is np.ndarray:
        a = a.tolist()
        b = b.tolist()

    mdiff, hull = convexHull(a, b)
    normvec = np.linalg.norm(hull, axis=1)

    if 0 in np.round(normvec, 2):
        return True
    elif np.round(subtended_angle(hull, [0, 0]), 2) == 360.0:
        return True
    else:
        return False


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
