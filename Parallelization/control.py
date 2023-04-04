import numpy as np
from animate import animation_run
from GJK2D import collison_check
import time


class Body(object):
    def __init__(self, state):
        self.state = state
        self.mass = 100.0  # kg
        self.inertia = 100.0  # kgm^2

    @property
    def pos(self):
        return self.state[0:2]

    @property
    def vel(self):
        return self.state[2:4]

    @property
    def theta(self):
        return self.state[4]

    @property
    def w(self):
        return self.state[5]


def Target_dynamics(dt, t, Target):
    pos = Target.pos
    vel = Target.vel
    theta = Target.theta
    w = Target.w

    posdot = vel
    veldot = np.zeros(2)
    thetadot = w
    wdot = 0.0

    statedot = np.hstack((posdot, veldot, thetadot, wdot))

    return statedot


def Chaser_dynamics(dt, t, Chaser):
    pos = Chaser.pos
    vel = Chaser.vel
    theta = Chaser.theta
    w = Chaser.w

    posdot = vel
    veldot = np.array([0.0, 0.0])
    thetadot = w

    Kp = 1.0
    Kv = 1.0
    wdot = -Kp*(theta - np.deg2rad(180)) + -Kv*w

    statedot = np.hstack((posdot, veldot, thetadot, wdot))

    return statedot


Target = Body(np.array([0.0, 0.0,
                        0.0, 0.0,
                        0.0, 0.0]))

Chaser = Body(np.array([4.0, 0.0,
                        -0.3, 0.0,
                        np.deg2rad(90.0), np.deg2rad(20.0)]))

t = 0
dt = 0.001
TMAX = 20.0

time_array = []
Target_state_array = []
Chaser_state_array = []

t1 = time.perf_counter()
while t < TMAX:

    Target.state = Target.state + dt*Target_dynamics(dt, t, Target)
    Chaser.state = Chaser.state + dt*Chaser_dynamics(dt, t, Chaser)
    t = t + dt

    time_array.append(t)
    Target_state_array.append(Target.state)
    Chaser_state_array.append(Chaser.state)

time_array = np.asarray(time_array)
Target_state_array = np.asarray(Target_state_array)
Chaser_state_array = np.asarray(Chaser_state_array)

t2 = time.perf_counter()
print("Dynamics Time Taken : ", t2-t1)

#Numba dummy run
t_dum = collison_check(time_array[0:1],
                       Target_state_array[0:1, :],
                       Chaser_state_array[0:1, :], dummy=1)

td = time.perf_counter()
print("Numba Compilation Time Taken : ", td-t2)
print(" ")

t_anim = collison_check(time_array, Target_state_array, Chaser_state_array)

t3 = time.perf_counter()
print(" ")
print("Col. Detect Time Taken : ", t3-td)

# ani = animation_run(time_array, dt, t_anim,
                    # Target_state_array, Chaser_state_array)
# figure_at_time(17.3, time_array, Target_state_array, Chaser_state_array)
