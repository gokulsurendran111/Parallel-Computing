import numpy as np
from ObjectDefinition import body1, body2
from numba import njit
import scipy.spatial as sp


def collison_check(time_array, Target_state_array, Chaser_state_array,
                   Target_Num, Chaser_Num):

    closest_dist_array = np.inf*np.ones(time_array.size)
    indices_B1_B2_array = -1*np.ones((time_array.size, 4), dtype='int')

    for i in range(time_array.size):

        t = time_array[i]
        closest_distance = np.inf
        # print("Solving For Time ", t)

        CG_b1 = Target_state_array[i, 0:3]
        q_b1 = Target_state_array[i, 6:10]
        CG_b2 = Chaser_state_array[i, 0:3]
        q_b2 = Chaser_state_array[i, 6:10]

        B1_points_ECI, B1_N = body1(CG_b1, q_b1, Target_Num)
        B2_points_ECI, B2_N = body2(CG_b2, q_b2, Chaser_Num)

        for ii in range(0, Target_Num):
            Target_shape = B1_points_ECI[ii, 0:B1_N[ii], :]
            for jj in range(0, Chaser_Num):
                Chaser_shape = B2_points_ECI[jj, 0:B2_N[jj], :]

                out, distance_info = isCollide(Target_shape, Chaser_shape)
                distance_SB1_SB2 = distance_info[0]
                ind_point_SB1 = int(distance_info[1])
                ind_point_SB2 = int(distance_info[2])

                if distance_SB1_SB2 <= closest_distance:
                    closest_distance = distance_SB1_SB2
                    indices_B1_B2 = np.array([ii, jj, ind_point_SB1,
                                              ind_point_SB2])

                if out == 1:
                    closest_dist_array[i] = 0.0
                    indices_B1_B2_array[i] = indices_B1_B2
                    return t, closest_dist_array, indices_B1_B2_array

        closest_dist_array[i] = closest_distance
        indices_B1_B2_array[i] = indices_B1_B2

    if i == time_array.size-1:
        return np.inf, closest_dist_array, indices_B1_B2_array


# a and b are shapes' coordinates
@njit(cache=True)
def minkowskiDiff(a, b):  # a and b denote points in the 2 bodies
    a_len = a.shape[0]
    b_len = b.shape[0]

    mDiff = np.zeros((a_len*b_len, 3))
    dist_array = np.zeros((a_len*b_len, 3))

    k = 0
    for i in range(a_len):
        for j in range(b_len):
            mDiff[k, 0] = a[i, 0] - b[j, 0]
            mDiff[k, 1] = a[i, 1] - b[j, 1]
            mDiff[k, 2] = a[i, 2] - b[j, 2]
            dist_from_origin = np.linalg.norm(mDiff[k, :])
            dist_info = np.array([dist_from_origin, i, j])
            dist_array[k] = dist_info
            k += 1

    smallest_dist_index = np.argmin(dist_array[:, 0])
    closest_dist_info = dist_array[smallest_dist_index, :]

    return mDiff, closest_dist_info


@njit(cache=True)
def subtended_angle(points, hull_triangles):

    Totang_solid = 0.0

    for k in range(0, hull_triangles.shape[0]):
        hull_points_array = points[hull_triangles[k]]
        a = hull_points_array[0]
        b = hull_points_array[1]
        c = hull_points_array[2]

        anorm = np.linalg.norm(a)
        bnorm = np.linalg.norm(b)
        cnorm = np.linalg.norm(c)

        numr = np.abs(np.dot(a, np.cross(b, c)))
        denr = anorm*bnorm*cnorm + np.dot(a, b)*cnorm + \
            np.dot(a, c)*bnorm + np.dot(b, c)*anorm
        omega = np.abs(2*np.arctan2(numr, denr))

        Totang_solid = Totang_solid + omega

    return Totang_solid


def isCollide(a, b):
    mDiff, closest_dist_info = minkowskiDiff(a, b)
    hull = sp.ConvexHull(mDiff)
    Totang_solid = subtended_angle(hull.points, hull.simplices)

    normvec = np.zeros(hull.points.shape[0])
    for i in range(normvec.size):
        normvec[i] = np.linalg.norm(hull.points)

    if normvec[normvec < 1e-3].size > 0:
        return 1, closest_dist_info
    elif np.abs((180/np.pi)*Totang_solid - 720.0) < 1e-3:
        return 1, closest_dist_info
    else:
        return 0, closest_dist_info
