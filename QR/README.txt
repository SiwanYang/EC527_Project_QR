/************************************/
High Performance Programming for QR Decomposition

This is a 3 people project

My part is the Baseline Implementation, which is used to compare to GPU version and pThread version.

My groupmates implements the optimization version (GPU and multi-thread) based on my code

/************************************/
Functionality:

Matrix decomposition is a process of factorizing matrix into a product of two matrices.

QR Decomposition is among the most commonly applied methods in matrix decomposition.

I implemented the Givens rotation and Householder reflection algorithms in one program to do QR decomposition. 

/************************************/
Compile and run

Source files are located in CPU_code folder. 

Run "make" command in /.../CPU_code/ directory 

Run "./qr <parameter>" to do QR decomposition on a matrix. The dimension of the matrix is determined by <parameter>, which is an integer(0-1000)

The running time will be given when the program is done.
