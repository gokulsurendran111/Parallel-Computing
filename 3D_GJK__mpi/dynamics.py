import numpy as np
import math_aux as ma
from numba import njit


class Body(object):
    def __init__(self, state, NumConvxObjects):
        self.state = state
        self.mass = 100.0  # kg
        self.inertia = 100.0  # kgm^2
        self.NumConvxObjects = NumConvxObjects

    @property
    def pos(self):
        return self.state[0:3]

    @property
    def vel(self):
        return self.state[3:6]

    @property
    def q(self):
        return self.state[6:10]

    @property
    def w(self):
        return self.state[10:13]


def Target_dynamics(dt, t, Target):
    pos = Target.pos
    vel = Target.vel
    q = Target.q
    w = Target.w

    posdot = vel
    veldot = np.zeros(3)
    qdot = 0.5*ma.quatmultiply(q, np.r_[0.0, w])
    wdot = np.zeros(3)

    statedot = np.hstack((posdot, veldot, qdot, wdot))

    return statedot


def Chaser_dynamics(dt, t, Chaser):
    pos = Chaser.pos
    vel = Chaser.vel
    q = Chaser.q
    w = Chaser.w

    posdot = vel
    veldot = np.zeros(3)
    qdot = 0.5*ma.quatmultiply(q, np.r_[0.0, w])

    # Kp = 2.0
    # Kv = 5.0
    # wdot = -Kp*(theta - np.deg2rad(180)) + -Kv*w
    wdot = np.zeros(3)

    statedot = np.hstack((posdot, veldot, qdot, wdot))

    return statedot


# Convention : State[Position, Velocity, Quaternion, Angular Velocity],
#              Number of Convex objects


def Simulate(Target, Chaser, t, dt, TMAX):
    time_array = []
    Target_state_array = []
    Chaser_state_array = []

    while t < TMAX:

        Target.state = Target.state + dt*Target_dynamics(dt, t, Target)
        Chaser.state = Chaser.state + dt*Chaser_dynamics(dt, t, Chaser)

        Target.state[6:10] = Target.state[6:10]/np.linalg.norm(Target.state[6:10])
        Chaser.state[6:10] = Chaser.state[6:10]/np.linalg.norm(Chaser.state[6:10])

        t = t + dt

        time_array.append(t)
        Target_state_array.append(Target.state)
        Chaser_state_array.append(Chaser.state)

    time_array = np.asarray(time_array)
    Target_state_array = np.asarray(Target_state_array)
    Chaser_state_array = np.asarray(Chaser_state_array)

    return time_array, Target_state_array, Chaser_state_array
