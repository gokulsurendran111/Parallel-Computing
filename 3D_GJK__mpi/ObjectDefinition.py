import numpy as np
from numba import njit
from math_aux import *


@njit(cache=True)
def body1(CG, q_ECI, NumConvxObjects):

    CGx = CG[0]
    CGy = CG[1]
    CGz = CG[2]

    N = np.zeros(NumConvxObjects, dtype='int')
    N[0] = 9

    Nmax = N.max()
    points = np.zeros((NumConvxObjects, Nmax, 3))

    # Cube with Pyramid on top (Local Frame)
    xl = -0.5 + np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5])
    yl = -0.5 + np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5])
    zl = -0.5 + np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.5])

    points[0, 0:N[0], :] = np.vstack((xl, yl, zl)).T

    # Transformation of vertex points to Inertial frame

    newpoints = np.zeros_like(points, dtype='double')
    for obj in range(NumConvxObjects):
        for ip in range(points.shape[1]):
            newpoints[obj, ip] = quatrotate(q_ECI, points[obj, ip])

    # Adding CG to the vertex points

    Transf_points = np.zeros_like(newpoints)

    Transf_points[:, :, 0] = CGx + newpoints[:, :, 0]
    Transf_points[:, :, 1] = CGy + newpoints[:, :, 1]
    Transf_points[:, :, 2] = CGz + newpoints[:, :, 2]

    return Transf_points, N


@njit(cache=True)
def body2(CG, q_ECI, NumConvxObjects):

    CGx = CG[0]
    CGy = CG[1]
    CGz = CG[2]

    N = np.zeros(NumConvxObjects, dtype='int')

    points_body1 = Cylinder(Radius=0.7, Height=1.0, NRadius=4, NCirc=20,
                            NHeight=4, CG_loc=[0.0, 0.0, 0.0])
    N[0] = points_body1.shape[0]

    points_body2 = Cylinder(Radius=0.1, Height=2.0, NRadius=1, NCirc=10,
                            NHeight=10, CG_loc=[0.0, 0.0, 1.5])
    N[1] = points_body2.shape[0]

    Nmax = N.max()
    points = np.zeros((NumConvxObjects, Nmax, 3))

    points[0, 0:N[0], :] = points_body1
    points[1, 0:N[1], :] = points_body2

    # -------------------------------------------------
    # Transformation of vertex points to Inertial frame

    newpoints = np.zeros_like(points, dtype='double')
    for obj in range(NumConvxObjects):
        for ip in range(points.shape[1]):
            newpoints[obj, ip] = quatrotate(q_ECI, points[obj, ip])

    # Adding CG to the vertex points

    Transf_points = np.zeros_like(newpoints)

    Transf_points[:, :, 0] = CGx + newpoints[:, :, 0]
    Transf_points[:, :, 1] = CGy + newpoints[:, :, 1]
    Transf_points[:, :, 2] = CGz + newpoints[:, :, 2]

    return Transf_points, N


@njit(cache=True)
def Cylinder(Radius, Height, NRadius, NCirc, NHeight, CG_loc):

    points = np.zeros((NCirc*NHeight + 2*NRadius*NCirc, 3))
    k = 0
    for h in np.linspace(-Height/2, Height/2, NHeight):
        t = np.linspace(0.0, 2*np.pi, NCirc)
        x = CG_loc[0] + Radius*np.cos(t)
        y = CG_loc[1] + Radius*np.sin(t)
        z = CG_loc[2] + h + np.zeros_like(x)
        points[k*NCirc:(k+1)*NCirc, :] = np.vstack((x, y, z)).T
        k += 1

    if NRadius > 1:
        delta = Radius/NRadius
        for r in np.linspace(delta, Radius-delta, NRadius-1):
            t = np.linspace(0.0, 2*np.pi, NCirc)
            x = CG_loc[0] + r*np.cos(t)
            y = CG_loc[1] + r*np.sin(t)
            z = CG_loc[2] + Height/2 + np.zeros_like(x)
            points[k*NCirc:(k+1)*NCirc, :] = np.vstack((x, y, z)).T
            k += 1

        for r in np.linspace(delta, Radius-delta, NRadius-1):
            t = np.linspace(0.0, 2*np.pi, NCirc)
            x = CG_loc[0] + r*np.cos(t)
            y = CG_loc[1] + r*np.sin(t)
            z = CG_loc[2] + -Height/2 + np.zeros_like(x)
            points[k*NCirc:(k+1)*NCirc, :] = np.vstack((x, y, z)).T
            k += 1

    return points[:k*NCirc, :]


def SeeBody():
    cg = np.zeros(3)
    qeci = np.zeros(4)
    qeci[0] = 1.0
    B2_points_ECI, B2_N = body1(cg, qeci, 1)
    print(B2_N)

    B2 = []
    B2p = []
    for obj in range(B2_N.size):
        hull = ConvexHull(B2_points_ECI[obj, 0:B2_N[obj]])
        x, y, z = np.split(B2_points_ECI[obj, 0:B2_N[obj]], 3, axis=1)
        B2.append(mlab.triangular_mesh(x, y, z, hull.simplices,
                                       colormap='YlGn'))
        B2p.append(mlab.points3d(x, y, z, scale_factor=0.05))

    mlab.show()


if __name__ == '__main__':
    from mayavi import mlab
    from scipy.spatial import ConvexHull
    SeeBody()
# --------------------------
# Object Design Dictionary
# --------------------------

# Cone Design
# n = 10
# t = np.linspace(0, 2*np.pi, n)
# x = 0.5 + 0.5*np.cos(t)
# y = 0.5 + 0.5*np.sin(t)
# z = 1.0 + np.zeros_like(x)

# triangles = [(0, i, i + 1) for i in range(1, n)]
# x = np.r_[0.5, x]
# y = np.r_[0.5, y]
# z = np.r_[1.5, z]
# mlab.triangular_mesh(x, y, z, triangles)
# mlab.points3d(x, y, z, scale_factor=0.1)

# ## Cube Design
# x = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
# y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
# z = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

# n = 8
# points = [[x[i], y[i], z[i]] for i in range(x.size)]
# hull = ConvexHull(points)

# triangles = [(i-1, i, i + 1) for i in range(1, n-1)]
# mlab.triangular_mesh(x, y, z, hull.simplices)
# mlab.points3d(x, y, z, scale_factor=0.1)

# ## Pyramid above Cube Design

# n = 5
# x = np.array([0.0, 0.0, 1.0, 1.0])
# y = np.array([0.0, 1.0, 0.0, 1.0])
# z = np.array([1.0, 1.0, 1.0, 1.0])

# x = np.r_[0.5, x]
# y = np.r_[0.5, y]
# z = np.r_[1.5, z]
# points = [[x[i], y[i], z[i]] for i in range(x.size)]

# hull = ConvexHull(points)
# mlab.triangular_mesh(x, y, z, hull.simplices)
# mlab.points3d(x, y, z, scale_factor=0.1)
