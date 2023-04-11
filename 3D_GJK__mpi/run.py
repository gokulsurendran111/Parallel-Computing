import numpy as np
import time
from animation import animate
from GJK3D import collison_check
from dynamics import Body, Simulate
from mpi4py import MPI
import sys


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


if rank == 0:
    NPROC = comm.Get_size()

    print(f"Running PROC = {rank} ...")

    Target = Body(np.array([0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0,
                            1.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, -0.4]), 1)

    Chaser = Body(np.array([-8.0, 0.0, 0.0,
                            0.25, 0.0, 0.0,
                            1.0, 0.0, 0.0, 0.0,
                            0.0, 0.2, -0.0]), 2)

    t = 0
    dt = 0.01
    TMAX = 50.0

    # Running Dynamics ------------------------
    t1 = time.perf_counter()
    time_ar, Target_states, Chaser_states = Simulate(Target, Chaser, t, dt, TMAX)
    t2 = time.perf_counter()
    print("Dynamics Time Taken : ", t2 - t1)

    # Running Collision Detection ---------------------

    proc_breaks = np.linspace(0, time_ar.size, NPROC+1, dtype='int')

    for irank in range(1, NPROC):
        i1 = proc_breaks[irank]
        i2 = proc_breaks[irank + 1]

        comm.send(i1, dest = irank, tag=10*irank + 0)
        comm.send(i2, dest = irank, tag=10*irank + 1)

        comm.Send(time_ar[i1:i2], dest = irank, tag=10*irank + 2)
        comm.Send(Target_states[i1:i2, :], dest = irank, tag=10*irank + 3)
        comm.Send(Chaser_states[i1:i2, :], dest = irank, tag=10*irank + 4)

        comm.send(Target.NumConvxObjects, dest = irank, tag=10*irank + 5)
        comm.send(Chaser.NumConvxObjects, dest = irank, tag=10*irank + 6)

    i1 = proc_breaks[0]
    i2 = proc_breaks[1]
    t_col = collison_check(time_ar[i1:i2], Target_states[i1:i2, :],
                            Chaser_states[i1:i2, :],
                            Target.NumConvxObjects, Chaser.NumConvxObjects)


    t_col_arr = []
    t_col_arr.append(t_col)

    for irank in range(1, NPROC):
        temp1 = comm.recv(source=irank, tag=irank)
        t_col_arr.append(temp1)

    t_anim = min(t_col_arr)
    if t_anim == np.inf:
        print('No Collision Detected')
    else:
        print(f'Collision Detected @ %5.2f s' % (t_anim))

    t3 = time.perf_counter()
    print(" ")
    print("Col. Detect Time Taken : ", t3 - t2)

    sys.stdout.flush()

    # Animation ---------------
    time_animation = time_ar[time_ar <= t_anim + dt]
    animate(time_animation, Target_states, Chaser_states,
            Target, Chaser)

else:
    print(f"Running PROC = {rank} ...")

    i1 = comm.recv(source = 0, tag=10*rank + 0)
    i2 = comm.recv(source = 0, tag=10*rank + 1)

    NTIMES = int(i2 - i1)
    time_ar = np.zeros(NTIMES)
    Target_states = np.zeros((NTIMES, 13))
    Chaser_states = np.zeros((NTIMES, 13))

    comm.Recv(time_ar, source = 0, tag=10*rank + 2)
    comm.Recv(Target_states, source = 0, tag=10*rank + 3)
    comm.Recv(Chaser_states, source = 0, tag=10*rank + 4)
    Target_NumConvx = comm.recv(source = 0, tag=10*rank + 5)
    Chaser_NumConvx = comm.recv(source = 0, tag=10*rank + 6)

    t_col = collison_check(time_ar, Target_states,
                            Chaser_states,
                            Target_NumConvx, Chaser_NumConvx)

    comm.send(t_col, dest = 0, tag=rank)

    # print(t_col)
    # print(time_ar[0])
    # print(Target_states[0, :])
    # print(Chaser_states[0, :])
    # print(Target_NumConvx)
    # print(Chaser_NumConvx)
