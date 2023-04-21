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

        comm.send(i1, dest = irank, tag=30*irank + 3)
        comm.send(i2, dest = irank, tag=30*irank + 4)

        comm.Send(time_ar[i1:i2], dest = irank, tag=30*irank + 5)
        comm.Send(Target_states[i1:i2, :], dest = irank, tag=30*irank + 6)
        comm.Send(Chaser_states[i1:i2, :], dest = irank, tag=30*irank + 7)

        comm.send(Target.NumConvxObjects, dest = irank, tag=30*irank + 8)
        comm.send(Chaser.NumConvxObjects, dest = irank, tag=30*irank + 9)

    i1 = proc_breaks[0]
    i2 = proc_breaks[1]
    t_col, clos_dist, clos_dist_info, flag_B1_B2, id_clos = collison_check(time_ar[i1:i2], 
                                                                   Target_states[i1:i2, :],
                                                                   Chaser_states[i1:i2, :],
                                                                   Target.NumConvxObjects, 
                                                                   Chaser.NumConvxObjects)


    t_col_arr = []
    t_col_arr.append(t_col)
    
    clos_dist_arr = []
    clos_dist_arr.append(clos_dist_arr)
    
    flag_B1_B2_arr = []
    flag_B1_B2_arr.append(np.floor(flag_B1_B2))
    
    id_clos_arr = []
    id_clos_arr.append(id_clos)
    
    clos_dist_info_arr = []
    clos_dist_info_arr.append(clos_dist_info)

    for irank in range(1, NPROC):
        temp1 = comm.recv(source=irank, tag=irank)
        t_col_arr.append(temp1)
        
        temp2 = comm.recv(source=irank, tag=irank+NPROC)
        clos_dist_arr.append(temp2)
        
        temp3 = np.zeros((1,2))
        comm.Recv(temp3, source = irank, tag=irank+2*NPROC)
        flag_B1_B2_arr.append(np.floor(temp3))
        
        temp4 = comm.recv(source=irank, tag=irank+3*NPROC)
        id_clos_arr.append(temp4)
        
        temp5 = np.zeros((temp4,3))
        comm.Recv(temp5, source = irank, tag=irank+4*NPROC)
        clos_dist_info_arr.append(temp5)

    
    t_anim = min(t_col_arr)
    id = t_col_arr.index(t_anim)
    
    if t_anim == np.inf:
        print('No Collision Detected')
        print(f'Closest Distance = %5.2f m' % (clos_dist_arr[id]))
    else:
        print(f'Collision Detected @ %5.2f s' % (t_anim))
        print("Final Closest Points index in ECI frame:")
        clos_dist_points = clos_dist_info_arr[id]
        sub_body = flag_B1_B2_arr[id]
        print("Points ", clos_dist_points[:,1], " in Body 1 - subbody ", sub_body[0][0]+1,
              " are close to points ", clos_dist_points[:,2], " in Body 2 - subbody ", 
              sub_body[0][1]+1, " respectively.")

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
    NPROC = comm.Get_size()
    i1 = comm.recv(source = 0, tag=30*rank + 3)
    i2 = comm.recv(source = 0, tag=30*rank + 4)

    NTIMES = int(i2 - i1)
    time_ar = np.zeros(NTIMES)
    Target_states = np.zeros((NTIMES, 13))
    Chaser_states = np.zeros((NTIMES, 13))

    comm.Recv(time_ar, source = 0, tag=30*rank + 5)
    comm.Recv(Target_states, source = 0, tag=30*rank + 6)
    comm.Recv(Chaser_states, source = 0, tag=30*rank + 7)
    Target_NumConvx = comm.recv(source = 0, tag=30*rank + 8)
    Chaser_NumConvx = comm.recv(source = 0, tag=30*rank + 9)

    t_col, clos_dist, clos_dist_info, flag_B1_B2, id_clos = collison_check(time_ar, 
                                                                  Target_states,
                                                                  Chaser_states,
                                                                  Target_NumConvx, 
                                                                  Chaser_NumConvx)

    # send_clos_info = clos_dist_info[:,1:]
    comm.send(t_col, dest = 0, tag=rank)
    comm.send(clos_dist, dest = 0, tag=rank + NPROC)
    comm.Send(flag_B1_B2, dest = 0, tag=rank + 2*NPROC)
    comm.send(id_clos, dest = 0, tag=rank + 3*NPROC)
    comm.Send(clos_dist_info, dest = 0, tag=rank + 4*NPROC)
   
