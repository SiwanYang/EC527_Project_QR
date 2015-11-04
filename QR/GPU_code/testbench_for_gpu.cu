#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define GIG 1000000000
#define PRINT_TIME 1
#define BLOCK 32 //blockDim
#define THREAD 32
#define SM_ARR_LEN 1024 //size of the matrix
#define THREADS_PER_BLOCK 64
#define THREADS_PER_DIM 16
#define TILING_DIM 20
#define MAX_MATRIX_SIZE_FOR_OUTPUT 64
#define QRD_SIMPLE_CALC_NORM2 128
#define QRD_SIMPLE_SCALE_COLUMN 128
#define QRD_SIMPLE_TRANSFORM_COLUMNS 128
#define QRD_OPTIMISED_NORM2_BLOCKSIZE 128
#define QRD_OPTIMISED_SCALE_COLUMN 128
#define QRD_OPTIMISED_TRANSFORM_COLUMNS_DIM 512
#define IMUL(a, b) __mul24(a, b)

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}



typedef struct
{
    float *n; // Length: size = width*height
    float *d; // Length: height
    unsigned int width;
    unsigned int height;
    unsigned int size;
    bool hasPivot;
    int *pivot; // Length: height
} QRmatrix;



__global__ void qrd_optimised_calc_norm2(float *out_norms, float *qr, int k, int m, int n)
{
	// Shared memory
	extern __shared__ float s_qr[];
    unsigned int r = blockIdx.x * blockDim.x + threadIdx.x + k; // Get row index
	// Clear cache for threads that exceeds max + they should not influence result
	s_qr[threadIdx.x] = 0;
	if (r < m)
	{
		// Read value to shared memory
		float val = qr[r * n + k];
		s_qr[threadIdx.x] = val * val;

		// Sync threads to make sure all other also have loaded values
		__syncthreads();

		// Do the actual pivot finding
		for(unsigned int stride = blockDim.x/2; stride>0; stride>>=1)
		{
			if (threadIdx.x < stride && (stride+threadIdx.x+k) < m)
			{
				s_qr[threadIdx.x] += s_qr[threadIdx.x + stride]; // Update value
			}

			// Sync threads
			__syncthreads();
		}
		// The first thread should write result from block to output
		if (threadIdx.x == 0)
		{
			out_norms[blockIdx.x] = s_qr[0]; // Load sum to output
		}
	}
}
__global__ void qrd_optimised_calc_norm2_L2(float *val_norms, int max)
{
	// Shared memory
	extern __shared__ float s_qr[];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// Clear cache for threads that exceeds max + they should not influence result
	s_qr[threadIdx.x] = 0;
	if (tid < max)
	{
		// Read value to shared memory
		s_qr[threadIdx.x] = val_norms[tid];
		// Sync threads to make sure all other also have loaded values
		__syncthreads();
		// Do the actual pivot finding
		for(unsigned int stride = blockDim.x/2; stride>0; stride>>=1)
		{
			if (threadIdx.x < stride)
			{
				s_qr[threadIdx.x] += s_qr[threadIdx.x + stride]; // Update value
			}

			// Sync threads
			__syncthreads();
		}
		// The first thread should write result from block to output
		if (threadIdx.x == 0)
		{
			val_norms[blockIdx.x] = s_qr[0]; // Load sum to output
		}
	}
}
__global__ void qrd_optimised_calc_norm2_FINISH(float *val_norms, float *qr, float *qr_diag, float *qr_norms, int k, int n)
{
	// Square root to get raw norm
	float nrm = sqrtf(val_norms[0]);

	// Flip sign for norm depending on the value of kth row kth column
	nrm = qr[k * n + k] < 0 ? -nrm : nrm;

	// Save the actual norm
	val_norms[0] = nrm;
	qr_norms[k] = nrm;

	// Flip sign for norm and save to QR diagonale
	qr_diag[k] = -nrm;
}


float calc_norm2_optimised(float *d_qr, float *d_diag, float *d_norms, int k, int m, int n)
{
	int threads = m-k;
    int blocks = (threads + QRD_OPTIMISED_NORM2_BLOCKSIZE-1) / QRD_OPTIMISED_NORM2_BLOCKSIZE;

	dim3 dimBlock(QRD_OPTIMISED_NORM2_BLOCKSIZE, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = QRD_OPTIMISED_NORM2_BLOCKSIZE*sizeof(float);

	float *d_out;

	cudaMalloc( (void**) &d_out, sizeof(float) * dimGrid.x);

	// First run on a, subsequential will be on d_out
	qrd_optimised_calc_norm2<<< dimGrid, dimBlock, smemSize >>>(d_out, d_qr, k, m, n);

	while(blocks > 1)
	{
		// Adjust the number of required blocks, for the second round
		threads = blocks;
		blocks = threads > QRD_OPTIMISED_NORM2_BLOCKSIZE
			? (threads + QRD_OPTIMISED_NORM2_BLOCKSIZE-1) / QRD_OPTIMISED_NORM2_BLOCKSIZE
			: 1;

		dimGrid.x = blocks;	
		qrd_optimised_calc_norm2_L2<<< dimGrid, dimBlock, smemSize >>>(d_out, threads);
	}
	
	qrd_optimised_calc_norm2_FINISH<<< 1, 1 >>>(d_out, d_qr, d_diag, d_norms, k, n);
	cudaThreadSynchronize();
	float norm = 0.0;
	cudaMemcpy( &norm, d_out, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree( d_out );
	return norm;
}

void output_matrix(float *m, int height, int width) {

    int size = height*width;
    if (size > MAX_MATRIX_SIZE_FOR_OUTPUT) 
	{
        printf("Matrix to big to be outputted...");
        printf("M: %dx%d (%d)\n", height, width, size);
        return;
    }
    
    int row = 0;
    
    for(int i = 0; i < size; i++) 
	{
        int curRow = i/width;
        //int curCol = i%m->width;
        if (curRow > row) 
		{
            printf("\n");
            row = curRow;
        }
        else if (i > 0) 
		{
            printf("  ");
        }       
        printf("%f", m[i] );
    }  
    printf("\n");
}

void output_qr_matrix(QRmatrix *m)
{
	output_matrix(m->n, m->height, m->width);
	printf("\nD:\n");
	if (m->size > MAX_MATRIX_SIZE_FOR_OUTPUT)
	{
		printf("Total diagonale elements %d.", m->height);
	}
	else
	{
		for(unsigned int i = 0; i < m->height; i++)
		{
			if (i > 0) printf(", ");
			printf("%f", m->d[i] );
		}
	}
	printf("\n");
}


__global__ void qrd_optimised_scale_column(float *qr, float *qr_norms, int k, int m, int n)
{
	float norm = qr_norms[k];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int r = k + tid;
	if (r < m)
	{
		qr[r * n + k] /= norm;
	}
	if (tid == 0)
	{
		qr[k * n + k] += 1.0;
	}
}



__global__ void qrd_optimised_transform_columns(float *qr, int k, int m, int n)
{
	// Declare cache
    extern __shared__ float v[];

	// Apply transformation to remaining columns.
	int c = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
	if (c < n)
	{
		float s = 0.0, knk = qr[k * n + k];
		for (int i = k; i < m;)
        {
			// Guard for not exceeding height of QR matrix + load k column values to shared memory
			v[threadIdx.x] = i+threadIdx.x < m ? qr[(i+threadIdx.x) * n + k] : 0;
			__syncthreads();
			// For all k values loaded, calc s
			for(int vi = 0; vi < blockDim.y; vi++, i++)
				s += v[vi] * qr[i * n + c];
        }
		s = (-s) / knk;
		for (int i = k; i < m;)
        {
			// Guard for not exceeding height of QR matrix + load k column values to shared memory
			v[threadIdx.x] = i+threadIdx.x < m ? qr[(i+threadIdx.x) * n + k] : 0;
			__syncthreads();
			// For all k values loaded, calc s
			for(int vi = 0; vi < blockDim.y; vi++, i++)
				qr[i * n + c] += s * v[vi];
        }
	}
}


void scale_column_optimised(float *d_qr, float *d_norms, int k, int m, int n)
{
	int threads = m-k;
    int blocks = (threads + QRD_OPTIMISED_SCALE_COLUMN-1) / QRD_OPTIMISED_SCALE_COLUMN;
	qrd_optimised_scale_column<<< blocks, QRD_OPTIMISED_SCALE_COLUMN >>>( d_qr, d_norms, k, m, n );
}



void transform_columns_optimised(float *d_qr, int k, int m, int n)
{
	int threads = n-(k+1);
    int blocks = (threads + QRD_OPTIMISED_TRANSFORM_COLUMNS_DIM-1) / QRD_OPTIMISED_TRANSFORM_COLUMNS_DIM;
	dim3 dimBlock(QRD_OPTIMISED_TRANSFORM_COLUMNS_DIM, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = QRD_OPTIMISED_TRANSFORM_COLUMNS_DIM*sizeof(float);
	qrd_optimised_transform_columns<<< dimGrid, dimBlock, smemSize >>>( d_qr, k, m, n );
}

void randomInit(float* data, int size, float min, float max)
{   
    const float range = (max - min);    
    for (int i = 0; i < size; ++i) 
	{
        float rnd = rand()/(float)RAND_MAX; // Generate random value from 0.0 to 1.0
        data[i] = rnd*range + min;
    }
}



QRmatrix* gpu_qrd_optimised(QRmatrix *a, int blockDimension, int version, bool usePivot) 
{
  // GPU Timing variables
	cudaEvent_t start, stop;
	float elapsed_gpu;
	bool output = a->size < 50;
	int n = a->width;
	int m = a->height;
    
    QRmatrix *qr = (QRmatrix*) malloc (sizeof(QRmatrix));
    qr->height = a->height;
    qr->size = a->size;
    qr->width = a->width;
    qr->n = (float*) malloc (sizeof(float)*a->size);
    qr->d = (float*) malloc (sizeof(float)*a->height);
    qr->pivot = (int*) malloc (sizeof(int)*a->height);
    qr->hasPivot = false;

	// Check if block dimension is too big
	if (blockDimension > (int)a->width) blockDimension = a->width;


    // Declare kernel pointers
    float *d_qr, *d_diag, *d_norms; // QR + Diagonale + Norms
    
    // Allocate memory on GPU for LU matrix
    cudaMalloc( (void**) &d_qr, sizeof(float) * qr->size );
    cudaMalloc( (void**) &d_diag, sizeof(float) * qr->width );
    cudaMalloc( (void**) &d_norms, sizeof(float) * qr->width );
	

    // Copy matrix A vector values to the QR pointer on the GPU/device = A copy to QR (will be modified)
    cudaMemcpy( d_qr, a->n, sizeof(float) * a->size, cudaMemcpyHostToDevice);

    #if PRINT_TIME
	// Create the cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Record event on the default stream
	cudaEventRecord(start, 0);
	#endif
	// Execute kernels
	// Optimised QR
	// Core algorithm for QR Decomposition
	for (int k = 0; k < n; k++) // For each column
    {
        // Compute 2-norm of k-th column
		float nrm = calc_norm2_optimised(d_qr, d_diag, d_norms, k, m, n);
        if (nrm != 0.0)
        {
            // Form k-th Householder vector. (Performed on device)
			scale_column_optimised(d_qr, d_norms, k, m, n);

			if (output)
			{
				cudaMemcpy( qr->n, d_qr, qr->size * sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy( qr->d, d_diag, qr->width * sizeof(float), cudaMemcpyDeviceToHost);
				printf("\n\nDecompose Partly Column=%d\n--------------\n", k);
				output_qr_matrix(qr);
			}
            // Apply transformation to remaining columns.
			transform_columns_optimised(d_qr, k, m, n);
        }

		if (output)
		{
			cudaMemcpy( qr->n, d_qr, qr->size * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy( qr->d, d_diag, qr->width * sizeof(float), cudaMemcpyDeviceToHost);	
			printf("\n\nDecompose Partly Column=%d\n--------------\n", k);
			output_qr_matrix(qr);
		}
    }
    cudaThreadSynchronize();
	#if PRINT_TIME
	// Stop and destroy the timer
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);
	printf("\nGPU time: %f (msec)\n", elapsed_gpu);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	#endif
    
    // Copy matrix members + value vector to host;
	cudaMemcpy( qr->n, d_qr, qr->size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy( qr->d, d_diag, qr->width * sizeof(float), cudaMemcpyDeviceToHost);

	// Release memory on device
    cudaFree( d_qr );
	cudaFree( d_diag );
	cudaFree( d_norms );
    cudaThreadExit();
    return qr;
}



void run_cuda_qr_decomposition(int aHeight, int aWidth, int type, int blockDimension, bool cpu_calc)
{
    // Declare matrices and allocate memory
    QRmatrix *a = (QRmatrix*) malloc (sizeof(QRmatrix));
    a->size = aHeight*aWidth;
    a->width = aWidth;
    a->height = aHeight;
    a->n = (float*) malloc (sizeof(float)*a->size);
    a->d = (float*) malloc (sizeof(float)*a->height);
    a->hasPivot = false;
	
	// Allocate memory for vector values
    randomInit(a->n, a->size, 0, 9);
    printf("Running QR Decomposition\n\n");
    printf("A: %dx%d (%d)\n", a->height, a->width, a->size);
    output_matrix(a->n, a->height, a->width);
    printf("\n");
        
    // Run matrix multiplication on GPU
    printf("Running on GPU...");
    QRmatrix *qr;
	bool pivot = false;
	qr = gpu_qrd_optimised(a, blockDimension, 1, false);
}




int main(int argc, char **argv)
{
    
    int type = 1;
    bool cpu = true;
  
	// Arrays on GPU global memory
	float *d_a, *d_b, *d_result;

	// Arrays on the host memory
	float *h_a, *h_b, *h_result, *h_judge;

	int arrLen = SM_ARR_LEN;

	printf("Length of the array = %d\n", arrLen);

	// Allocate GPU memory
	size_t allocSize = arrLen * arrLen * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_a, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_b, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_result, allocSize));
    
	// Allocate arrays on host memory
	h_a                    = (float *) malloc(allocSize);
	h_b                    = (float *) malloc(allocSize);
	h_result               = (float *) malloc(allocSize);
	h_judge                = (float *) malloc(allocSize);
	// Transfer the arrays to the GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, allocSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, allocSize, cudaMemcpyHostToDevice));
	dim3 globalGrid(BLOCK, BLOCK, 1);
	dim3 globalBlock(THREAD, THREAD, 1);	    
	run_cuda_qr_decomposition(SM_ARR_LEN, SM_ARR_LEN, type, 20, cpu);
	cudaDeviceSynchronize();
	// Check for errors during launch
	CUDA_SAFE_CALL(cudaPeekAtLastError());
  return 0;
}

