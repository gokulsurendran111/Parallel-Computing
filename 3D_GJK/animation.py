import numpy as np
from mayavi import mlab
from scipy.spatial import ConvexHull
from math_aux import *

def body1(CG, q_ECI, init=0):

    CGx = CG[0]
    CGy = CG[1]
    CGz = CG[2]

    # Cube with Pyramid on top - Design
    xl = -0.5 + np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5])
    yl = -0.5 + np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5])
    zl = -0.5 + np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.5])

    points = np.array([xl, yl, zl])
    newpoints = np.zeros_like(points, dtype='double')
    for ip in range(points.shape[1]):
        newpoints.T[ip] = quatrotate(q_ECI, points.T[ip])

    xlr = newpoints[0]
    ylr = newpoints[1]
    zlr = newpoints[2]

    x = CGx + xlr
    y = CGy + ylr
    z = CGz + zlr

    points = [[x[i], y[i], z[i]] for i in range(x.size)]
    hull = ConvexHull(points)

    if init == 1:
        B1 = mlab.triangular_mesh(x, y, z, hull.simplices, colormap='jet')
        B1p = mlab.points3d(x, y, z, scale_factor=0.1)
        return B1, B1p
    else:
        return x, y, z


def body2(CG, q_ECI, init=0):

    CGx = CG[0]
    CGy = CG[1]
    CGz = CG[2]

    # Cylinder
    n = 20
    t = np.linspace(0, 2*np.pi, n)

    xdown = 0.7*np.cos(t)
    ydown = 0.7*np.sin(t)
    zdown = -0.5 + np.zeros_like(xdown)

    xup = 0.3*np.cos(t)
    yup = 0.3*np.sin(t)
    zup = 0.5 + np.zeros_like(xup)

    xl = np.concatenate((xdown, xup))
    yl = np.concatenate((ydown, yup))
    zl = np.concatenate((zdown, zup))

    points = np.array([xl, yl, zl])
    newpoints = np.zeros_like(points, dtype='double')
    for ip in range(points.shape[1]):
        newpoints.T[ip] = quatrotate(q_ECI, points.T[ip])

    xlr = newpoints[0]
    ylr = newpoints[1]
    zlr = newpoints[2]

    x = CGx + xlr
    y = CGy + ylr
    z = CGz + zlr

    points = [[x[i], y[i], z[i]] for i in range(x.size)]

    if init == 1:
        hull = ConvexHull(points)
        B2 = mlab.triangular_mesh(x, y, z, hull.simplices)
        B2p = mlab.points3d(x, y, z, scale_factor=0.1)
        return B2, B2p
    else:
        return x, y, z


def animate(t_array, Target_array, Chaser_array):
    mlab.figure(size=(800, 600), bgcolor=(0, 0, 0))
    mlab.clf()

    CG_b1 = Target_array[0, 0:3]
    q_b1 = Target_array[0, 6:10]
    CG_b2 = Chaser_array[0, 0:3]
    q_b2 = Chaser_array[0, 6:10]

    B1, B1p = body1(CG_b1, q_b1, init=1)
    B2, B2p = body2(CG_b2, q_b2, init=1)

    @mlab.animate(delay=10, ui=True)
    def anim():
        for i in range(t_array.size):
            CG_b1 = Target_array[i, 0:3]
            q_b1 = Target_array[i, 6:10]

            CG_b2 = Chaser_array[i, 0:3]
            q_b2 = Chaser_array[i, 6:10]

            x, y, z = body1(CG_b1, q_b1)
            B1.mlab_source.trait_set(x=x, y=y, z=z)
            B1p.mlab_source.trait_set(x=x, y=y, z=z)

            x, y, z = body2(CG_b2, q_b2)
            B2.mlab_source.trait_set(x=x, y=y, z=z)
            B2p.mlab_source.trait_set(x=x, y=y, z=z)
            yield

    anim()
    mlab.show()

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
