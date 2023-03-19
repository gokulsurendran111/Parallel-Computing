import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


@njit
def Target_body(Target_pos, Target_theta):
    p1b = np.array([1.0, 1.0])
    p2b = np.array([1.0, -1.0])
    p3b = np.array([-1.0, -1.0])
    p4b = np.array([-1.0, 1.0])

    c1b = np.array([0.0, 0.0])
    c2b = np.array([1.0, 0.25])
    c3b = np.array([1.0, -0.25])

    R = np.eye(2)
    R[0, 0] = np.cos(Target_theta)
    R[0, 1] = -np.sin(Target_theta)
    R[1, 0] = np.sin(Target_theta)
    R[1, 1] = np.cos(Target_theta)

    p1 = Target_pos + R @ p1b
    p2 = Target_pos + R @ p2b
    p3 = Target_pos + R @ p3b
    p4 = Target_pos + R @ p4b

    c1 = Target_pos + R @ c1b
    c2 = Target_pos + R @ c2b
    c3 = Target_pos + R @ c3b

    Target_x_coord = np.array([p1[0], p2[0], p3[0], p4[0], p1[0],
                               c1[0], c2[0], c3[0], c1[0]])
    Target_y_coord = np.array([p1[1], p2[1], p3[1], p4[1], p1[1],
                               c1[1], c2[1], c3[1], c1[1]])

    return Target_x_coord, Target_y_coord


@njit
def Chaser_body(Chaser_pos, Chaser_theta):
    p1b = np.array([0.5, 0.5])
    p2b = np.array([0.5, -0.5])
    p3b = np.array([-0.5, -0.5])
    p4b = np.array([-0.5, 0.5])

    r1b = np.array([0.5, 0.0])
    r2b = np.array([1.5, 0.0])

    R = np.eye(2)
    R[0, 0] = np.cos(Chaser_theta)
    R[0, 1] = -np.sin(Chaser_theta)
    R[1, 0] = np.sin(Chaser_theta)
    R[1, 1] = np.cos(Chaser_theta)

    p1 = Chaser_pos + R @ p1b
    p2 = Chaser_pos + R @ p2b
    p3 = Chaser_pos + R @ p3b
    p4 = Chaser_pos + R @ p4b

    r1 = Chaser_pos + R @ r1b
    r2 = Chaser_pos + R @ r2b

    Chaser_x_coord = np.array([p1[0], p2[0], p3[0], p4[0], p1[0],
                               r1[0], r2[0]])
    Chaser_y_coord = np.array([p1[1], p2[1], p3[1], p4[1], p1[1],
                               r1[1], r2[1]])

    return Chaser_x_coord, Chaser_y_coord


# # Animation
def animation_run(time_array, dt, Target_state_array, Chaser_state_array):

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-2, 6), ylim=(-3, 3))
    ax.set_aspect('equal')
    ax.grid(True)

    target1, = ax.plot([], [], 'o-', lw=3)
    target2, = ax.plot([], [], 'o-', lw=1)
    chaser, = ax.plot([], [], 'o-', lw=3)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def anime(i):

        Tx_coord, Ty_coord = Target_body(Target_state_array[i, 0:2],
                                         Target_state_array[i, 4])
        Cx_coord, Cy_coord = Chaser_body(Chaser_state_array[i, 0:2],
                                         Chaser_state_array[i, 4])

        target1.set_data(Tx_coord[0:5], Ty_coord[0:5])
        target2.set_data(Tx_coord[5:9], Ty_coord[5:9])
        chaser.set_data(Cx_coord, Cy_coord)
        time_text.set_text(time_template % (i*dt))

        return target1, target2, chaser, time_text

    ani = FuncAnimation(fig, anime, time_array.size, interval=50, blit=True)
    plt.show()

    return ani
