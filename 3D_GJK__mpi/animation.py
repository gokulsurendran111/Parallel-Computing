import numpy as np
from mayavi import mlab
from ObjectDefinition import body1, body2
from scipy.spatial import ConvexHull


def animate(t_array, Target_array, Chaser_array, Target, Chaser):

    mlab.figure(size=(800, 600), bgcolor=(0, 0, 0))
    mlab.clf()

    CG_b1 = Target_array[0, 0:3]
    q_b1 = Target_array[0, 6:10]
    CG_b2 = Chaser_array[0, 0:3]
    q_b2 = Chaser_array[0, 6:10]

    B1_points_ECI, B1_N = body1(CG_b1, q_b1, Target.NumConvxObjects)
    B2_points_ECI, B2_N = body2(CG_b2, q_b2, Chaser.NumConvxObjects)

    B1 = []
    B1p = []
    for obj in range(B1_N.size):
        hull = ConvexHull(B1_points_ECI[obj, 0:B1_N[obj]])
        x, y, z = np.split(B1_points_ECI[obj, 0:B1_N[obj]], 3, axis=1)
        B1.append(mlab.triangular_mesh(x, y, z, hull.simplices))
        B1p.append(mlab.points3d(x, y, z, scale_factor=0.05))

    B2 = []
    B2p = []
    for obj in range(B2_N.size):
        hull = ConvexHull(B2_points_ECI[obj, 0:B2_N[obj]])
        x, y, z = np.split(B2_points_ECI[obj, 0:B2_N[obj]], 3, axis=1)
        B2.append(mlab.triangular_mesh(x, y, z, hull.simplices,
                                       colormap='YlGn'))
        B2p.append(mlab.points3d(x, y, z, scale_factor=0.05))

    @mlab.animate(delay=10, ui=True)
    def anim():
        tstep = int(0.1/(t_array[1] - t_array[0]))
        intervals = np.arange(0, t_array.size, tstep, dtype='int')
        for i in intervals:
            CG_b1 = Target_array[i, 0:3]
            q_b1 = Target_array[i, 6:10]

            CG_b2 = Chaser_array[i, 0:3]
            q_b2 = Chaser_array[i, 6:10]

            B1_points_ECI, B1_N = body1(CG_b1, q_b1, Target.NumConvxObjects)
            for obj in range(Target.NumConvxObjects):
                x, y, z = np.split(B1_points_ECI[obj, 0:B1_N[obj]], 3, axis=1)
                B1[obj].mlab_source.trait_set(x=x, y=y, z=z)
                B1p[obj].mlab_source.trait_set(x=x, y=y, z=z)
            yield

            B2_points_ECI, B2_N = body2(CG_b2, q_b2, Chaser.NumConvxObjects)
            for obj in range(Chaser.NumConvxObjects):
                x, y, z = np.split(B2_points_ECI[obj, 0:B2_N[obj]], 3, axis=1)
                B2[obj].mlab_source.trait_set(x=x, y=y, z=z)
                B2p[obj].mlab_source.trait_set(x=x, y=y, z=z)
            yield

    anim()
    mlab.show()
