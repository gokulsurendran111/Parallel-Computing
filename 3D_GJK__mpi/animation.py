import numpy as np
from mayavi import mlab
from ObjectDefinition import body1, body2
from scipy.spatial import ConvexHull


def animate(t_array, Target_array, Chaser_array, Target, Chaser,
            b1mark, b2mark, i_col_true):

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
        B1.append(mlab.triangular_mesh(x, y, z, hull.simplices,
                                       colormap='winter'))
        B1p.append(mlab.points3d(x, y, z, scale_factor=0.1))

    B2 = []
    B2p = []
    for obj in range(B2_N.size):
        hull = ConvexHull(B2_points_ECI[obj, 0:B2_N[obj]])
        x, y, z = np.split(B2_points_ECI[obj, 0:B2_N[obj]], 3, axis=1)
        B2.append(mlab.triangular_mesh(x, y, z, hull.simplices,
                                       colormap='Set3'))
        B2p.append(mlab.points3d(x, y, z, scale_factor=0.1))

    txt = mlab.text(0.6, 0.9, f'Time = {np.round(0.0, 2)}s', width=0.2)
    txt.property.bold = True
    txt.property.font_family = 'courier'
    txt.property.use_tight_bounding_box = True
    txt.property.font_size = 30
    txt.actor.text_scale_mode = 'none'

    @mlab.animate(delay=10, ui=True)
    def anim():
        tstep = int(0.1/(t_array[1] - t_array[0]))
        intervals = np.arange(0, t_array.size, tstep, dtype='int')
        intervals = np.r_[intervals, t_array.size-1]
        for i in intervals:
            CG_b1 = Target_array[i, 0:3]
            q_b1 = Target_array[i, 6:10]

            CG_b2 = Chaser_array[i, 0:3]
            q_b2 = Chaser_array[i, 6:10]

            txt.text = f'Time = {np.round(t_array[i], 2)}s'

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

        if i_col_true == 1:
            t_index = i
            CG_b1 = Target_array[t_index, 0:3]
            q_b1 = Target_array[t_index, 6:10]

            CG_b2 = Chaser_array[t_index, 0:3]
            q_b2 = Chaser_array[t_index, 6:10]

            B1_points_ECI, B1_N = body1(CG_b1, q_b1, Target.NumConvxObjects)
            B2_points_ECI, B2_N = body2(CG_b2, q_b2, Chaser.NumConvxObjects)

            x1, y1, z1 = np.split(B1_points_ECI[b1mark[0], b1mark[1]], 3)
            x2, y2, z2 = np.split(B2_points_ECI[b2mark[0], b2mark[1]], 3)
            mlab.points3d(x1, y1, z1, color=(1, 0, 0), scale_factor=0.1)
            mlab.points3d(x2, y2, z2, color=(1, 0, 0), scale_factor=0.1)

    anim()

    mlab.show()
