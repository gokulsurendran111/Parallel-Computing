import numpy as np
from ObjectDefinition import body1, body2
from numba import njit
import scipy.spatial as sp


def collison_check(time_array, Target_state_array, Chaser_state_array,
                   Target_Num, Chaser_Num):

    distance_B1_B2 = 1e8
    closest_dist_array = np.array([])
    
    for i in range(time_array.size):

        t = time_array[i]
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
                Chaser_shape = B2_points_ECI[jj, 0:B2_N[ii], :]

                out, closest_dist_info = isCollide(Target_shape, Chaser_shape)
                closest_distance = closest_dist_info[0, 0]
                flag_B1_B2 = np.array([ii, jj, t])
                
                if closest_distance <= distance_B1_B2:
                    distance_B1_B2 = closest_distance
                    
                if out == 1:
                    # print('Collision detected at ', t, ' s')
                    closest_dist_array = np.append(closest_dist_array, 
                                                   distance_B1_B2)
                    return t, closest_dist_array, closest_dist_info, flag_B1_B2, closest_dist_info.shape[0]

        closest_dist_array = np.append(closest_dist_array, distance_B1_B2)
    if i == time_array.size-1:
        # print('No Collision detected')
        return np.inf, closest_dist_array, closest_dist_info, flag_B1_B2, closest_dist_info.shape[0]


# a and b are shapes' coordinates
@njit(cache=True)
def minkowskiDiff(a, b):  # a and b denote points in the 2 bodies
    a_len = a.shape[0]
    b_len = b.shape[0]

    mDiff = np.zeros((a_len*b_len, 3))
    
    dist_array = np.array([0, 0, 0], dtype=np.float64).reshape(1,3)
    
    k = 0
    for i in range(a_len):
        for j in range(b_len):
            mDiff[k, 0] = a[i, 0] - b[j, 0]
            mDiff[k, 1] = a[i, 1] - b[j, 1]
            mDiff[k, 2] = a[i, 2] - b[j, 2]
            dist_from_origin = np.sqrt((mDiff[k, 0])**2 + (mDiff[k, 1])**2 + 
                                        (mDiff[k, 2])**2)
            dist_info = np.array([dist_from_origin, i, j]).reshape(1,3)
            dist_array = np.concatenate((dist_array,dist_info), axis=0)
            
            k += 1

    dist_array = dist_array[1:, :]

    temp = (dist_array[:, 0]).min()
    dist_index = np.array([0], dtype=np.int64)
    for i in range(dist_array.shape[0]):
        if dist_array[i, 0] == temp:
            dist_index = np.append(dist_index, i)
    dist_index = dist_index[1:]

    closest_dist_info = dist_array[dist_index, :]

    return mDiff, closest_dist_info
# for 3D bodies, mDiff is an array of size (a_len*b_len, 3)


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
