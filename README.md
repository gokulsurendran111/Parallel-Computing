# Parallel-Computing Project

A software to detect collision between two arbitrarily shaped bodies. We are using the GJK algorithm to detect collision. 
This project is still under development. 

--------------------------------------------------------
# Steps to execute 3D Collision detection code (Windows)

1. Download the code from the directory, 3D_GJK__mpi
2. Install mpi4py for Windows. Follow the this link, https://nyu-cds.github.io/python-mpi/setup/
3. Run the command `mpiexec -n 4 python run.py` to execute the code with 4 parallel MPI processes. 
