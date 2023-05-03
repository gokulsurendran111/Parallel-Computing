import numpy as np
import time
from animation import animate
from GJK3D import collison_check
from dynamics import Body, Simulate
from mpi4py import MPI
import sys
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


if rank == 0:
    NPROC = comm.Get_size()

    print(f"Running PROC = {rank} ...")

    f = open('Body1.txt')
    cg_b1 = np.array([float(x) for x in next(f).split()])
    vel_b1 = np.array([float(x) for x in next(f).split()])
    q_b1 = np.array([float(x) for x in next(f).split()])
    omeg_b1 = np.array([float(x) for x in next(f).split()])
    f.close()

    Tstate = np.hstack((cg_b1, vel_b1, q_b1, omeg_b1))
    Target = Body(Tstate, 2)

    f = open('Body2.txt')
    cg_b2 = np.array([float(x) for x in next(f).split()])
    vel_b2 = np.array([float(x) for x in next(f).split()])
    q_b2 = np.array([float(x) for x in next(f).split()])
    omeg_b2 = np.array([float(x) for x in next(f).split()])
    f.close()

    Cstate = np.hstack((cg_b2, vel_b2, q_b2, omeg_b2))
    Chaser = Body(Cstate, 3)

    f = open('simparam.txt')
    TMAX = np.array([float(x) for x in next(f).split()])
    f.close()

    t = 0
    dt = 0.01

    # Running Dynamics ------------------------
    t1 = time.perf_counter()
    time_ar, Target_states, Chaser_states = Simulate(Target, Chaser, t,
                                                     dt, TMAX)
    t2 = time.perf_counter()
    print("\n . . . Dynamics Simulation Completed . . . \n")
    print("Dynamics Time Taken : ", t2 - t1)
    print("\n . . . Collision Detection Started . . . \n")

    sys.stdout.flush()
    # Running Collision Detection ---------------------

    proc_breaks = np.linspace(0, time_ar.size, NPROC+1, dtype='int')

    for irank in range(1, NPROC):
        i1 = proc_breaks[irank]
        i2 = proc_breaks[irank + 1]

        comm.send(i1, dest=irank, tag=30*irank + 3)
        comm.send(i2, dest=irank, tag=30*irank + 4)

        comm.Send(time_ar[i1:i2], dest=irank, tag=30*irank + 5)
        comm.Send(Target_states[i1:i2, :], dest=irank, tag=30*irank + 6)
        comm.Send(Chaser_states[i1:i2, :], dest=irank, tag=30*irank + 7)

        comm.send(Target.NumConvxObjects, dest=irank, tag=30*irank + 8)
        comm.send(Chaser.NumConvxObjects, dest=irank, tag=30*irank + 9)

    i1 = proc_breaks[0]
    i2 = proc_breaks[1]
    t_col, clos_dist_hist, index_hist = collison_check(time_ar[i1:i2],
                                                       Target_states[i1:i2, :],
                                                       Chaser_states[i1:i2, :],
                                                       Target.NumConvxObjects,
                                                       Chaser.NumConvxObjects)

    t_col_arr = []
    t_col_arr.append(t_col)

    clos_dist_hist_arr = clos_dist_hist
    index_hist_arr = index_hist

    for irank in range(1, NPROC):
        i1 = proc_breaks[irank]
        i2 = proc_breaks[irank + 1]

        t_col_nproc = comm.recv(source=irank, tag=irank)
        t_col_arr.append(t_col_nproc)

        NTIMES = int(i2 - i1)
        clos_dist_nproc = np.zeros(NTIMES)
        comm.Recv(clos_dist_nproc, source=irank, tag=irank+NPROC)
        clos_dist_hist_arr = np.hstack((clos_dist_hist_arr, clos_dist_nproc))

        index_hist_nproc = np.zeros((NTIMES, 4), dtype='int')
        comm.Recv(index_hist_nproc, source=irank, tag=irank+2*NPROC)
        index_hist_arr = np.vstack((index_hist_arr, index_hist_nproc))

    t_anim = min(t_col_arr)
    id_closest_dis = np.argmin(clos_dist_hist_arr)

    print("\n . . . Collision Detection Completed . . . \n")

    t3 = time.perf_counter()
    print(" ")
    print("Col. Detect Time Taken : ", t3 - t2)

    b1_subbody = index_hist_arr[id_closest_dis, 0]
    b1_point = index_hist_arr[id_closest_dis, 2]

    b2_subbody = index_hist_arr[id_closest_dis, 1]
    b2_point = index_hist_arr[id_closest_dis, 3]

    if t_anim == np.inf:
        i_col_true = 0
        print('\n No Collision Detected')
        print(f'Closest Distance = %5.2f m' %
              (clos_dist_hist_arr[id_closest_dis]))
    else:
        i_col_true = 1
        print(f'\nCollision Detected @ %5.2f s' % (t_anim))
        print("ID of Collision Points :")
        print(f"Body 1 (Target) :: Sub body {b1_subbody}, Point {b1_point}")
        print(f"Body 2 (Chaser) :: Sub body {b2_subbody}, Point {b2_point}")

    sys.stdout.flush()

    # Animation ---------------
    time_animation = time_ar[time_ar <= t_anim]

    plot_len = time_animation.shape[0]
    plt.figure(1)
    plt.grid(True)
    plt.plot(time_animation, clos_dist_hist_arr[0:plot_len], 'b-', lw=2)
    plt.xlim([0.0, np.ceil(time_animation[-1])])
    plt.xlabel('Time (s)', fontsize=15)
    plt.ylabel('Closest Distance (m)', fontsize=15)
    plt.show()

    b1mark = [b1_subbody, b1_point]
    b2mark = [b2_subbody, b2_point]
    animate(time_animation, Target_states, Chaser_states,
            Target, Chaser, b1mark, b2mark, i_col_true)

else:
    print(f"Running PROC = {rank} ...")
    NPROC = comm.Get_size()
    i1 = comm.recv(source=0, tag=30*rank + 3)
    i2 = comm.recv(source=0, tag=30*rank + 4)

    NTIMES = int(i2 - i1)
    time_ar = np.zeros(NTIMES)
    Target_states = np.zeros((NTIMES, 13))
    Chaser_states = np.zeros((NTIMES, 13))

    comm.Recv(time_ar, source=0, tag=30*rank + 5)
    comm.Recv(Target_states, source=0, tag=30*rank + 6)
    comm.Recv(Chaser_states, source=0, tag=30*rank + 7)
    Target_NumConvx = comm.recv(source=0, tag=30*rank + 8)
    Chaser_NumConvx = comm.recv(source=0, tag=30*rank + 9)

    t_col, clos_dist_hist, index_hist = collison_check(time_ar,
                                                       Target_states,
                                                       Chaser_states,
                                                       Target_NumConvx,
                                                       Chaser_NumConvx)

    comm.send(t_col, dest=0, tag=rank)
    comm.Send(clos_dist_hist, dest=0, tag=rank + NPROC)
    comm.Send(index_hist, dest=0, tag=rank + 2*NPROC)
