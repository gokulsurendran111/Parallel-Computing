import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



# @njit
def Target_body(Target_pos, Target_theta):  # Pentagon

    N = 5
    points = [[np.cos(2*np.pi*k/N), np.sin(2*np.pi*k/N)] for k in range(N)]
    points.append(points[0])
    points = np.array(points)

    R = np.eye(2)
    R[0, 0] = np.cos(Target_theta)
    R[0, 1] = -np.sin(Target_theta)
    R[1, 0] = np.sin(Target_theta)
    R[1, 1] = np.cos(Target_theta)

    points_ECI = np.empty_like(points)
    for i in range(N+1):
        points_ECI[i] = Target_pos + R @ points[i]

    Target_x_coord = points_ECI[:, 0]
    Target_y_coord = points_ECI[:, 1]

    return Target_x_coord, Target_y_coord


# @njit
def Chaser_body(Chaser_pos, Chaser_theta):
    N = 4
    points = [[np.cos(2*np.pi*k/N), np.sin(2*np.pi*k/N)] for k in range(N)]
    points.append(points[0])
    points = np.array(points)

    R = np.eye(2)
    R[0, 0] = np.cos(Chaser_theta)
    R[0, 1] = -np.sin(Chaser_theta)
    R[1, 0] = np.sin(Chaser_theta)
    R[1, 1] = np.cos(Chaser_theta)

    points_ECI = np.empty_like(points)
    for i in range(N+1):
        points_ECI[i] = Chaser_pos + R @ points[i]

    Chaser_x_coord = points_ECI[:, 0]
    Chaser_y_coord = points_ECI[:, 1]

    return Chaser_x_coord, Chaser_y_coord

# # Animation


def animation_run(time_array, dt, t_anim,
                  Target_state_array, Chaser_state_array):

    index_final_anim = np.abs(time_array - t_anim).argmin() + 1

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-2, 6), ylim=(-3, 3))
    ax.set_aspect('equal')
    ax.grid(True)

    target1, = ax.plot([], [], 'o-', lw=3)
    chaser, = ax.plot([], [], 'o-', lw=3)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def anime(i):

        Tx_coord, Ty_coord = Target_body(Target_state_array[i, 0:2],
                                         Target_state_array[i, 4])
        Cx_coord, Cy_coord = Chaser_body(Chaser_state_array[i, 0:2],
                                         Chaser_state_array[i, 4])

        target1.set_data(Tx_coord, Ty_coord)
        chaser.set_data(Cx_coord, Cy_coord)
        time_text.set_text(time_template % (time_array[i]))

        return target1, chaser, time_text

    ani = FuncAnimation(fig, anime, index_final_anim,
                        interval=15, blit=True, repeat=False)
    plt.show()

    return ani


def figure_at_time(t_anim, time_array, Target_state_array, Chaser_state_array):

    i = np.abs(time_array - t_anim).argmin()

    t = time_array[i]
    Target_pos = Target_state_array[i, 0:2]
    Target_theta = Target_state_array[i, 4]
    Chaser_pos = Chaser_state_array[i, 0:2]
    Chaser_theta = Chaser_state_array[i, 4]

    Tx_coord, Ty_coord = Target_body(Target_pos, Target_theta)
    Cx_coord, Cy_coord = Chaser_body(Chaser_pos, Chaser_theta)

    plt.cla()
    plt.grid(True)
    plt.title('T = ' + str(round(t, 3)) + 's')
    plt.plot(Tx_coord, Ty_coord, 'bo-', lw=3)

    plt.plot(Cx_coord, Cy_coord, 'ro-', lw=3)
    plt.axis('equal')
    plt.xlim([-2, 6])
    plt.ylim([-3, 3])

    plt.show()
