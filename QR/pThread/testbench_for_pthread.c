/*****************************************************************************/
// gcc -O1 -fopenmp -o test testbench_for_pthread.c -lrt -lm


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define GIG 1000000000
#define OPTIONS 2
#define IDENT 0
#ifndef EPSLION
#define EPSLION 1.0e-10
#define M 1000
#define N 1000
#endif
#define NUM_THREADS 4

typedef double data_t;

/*struct used to hold all the thread data*/
struct thread_data{
	int thread_id;
	int width;
	int i;
	double alpha;
	double* a;
	double* u;
};

/*the function used to input a matrix*/
void Input_1(double A[M][N])
{
	int i,j;
	double temp;
	double a=1.0;
	for (i=0;i<M;i++)
	{
		for (j=0;j<N;j++)
		{
			A[i][j]=rand();
		}
	}
}
/*householder transformation*/
void householder (double A[M][N],double y[M],int i)
{
	int j,l;
	static double aa,ki,u[M],alpha;
	aa=0.0;
	for (l=i;l<M;l++) 
		aa=aa+A[l][i]*A[l][i];
	aa=sqrt(aa);
	if (A[i][i]>0.0) 
		ki=-aa;
	else 
		ki=aa;
	alpha=1.0/(ki*(ki-A[i][i]));
	u[i]=A[i][i]-ki;
	A[i][i]=ki;
	for(l=i+1;l<M;l++) 
		{
			u[l]=A[l][i];
			A[l][i]=0.0;
		}
	for(j=i+1;j<N;j++)
	{
		aa=0.0;
		for(l=i;l<M;l++) 
			aa=aa+u[l]*A[l][j];
		aa=alpha*aa;
		for(l=i;l<M;l++) 
			A[l][j]=A[l][j]-aa*u[l];
	}
	aa=0.0;
	for(l=i;l<M;l++) 
		aa=aa+u[l]*y[l];
	aa=alpha*aa;
	for(l=i;l<M;l++) 
		y[l]=y[l]-aa*u[l];
}

/*function performed in every pthread, the main part of householder*/
void *work(void *threadarg)
{
	int j, l;
	double aa;
	long int low, high;
	struct thread_data *my_data;
	my_data = (struct thread_data *) threadarg;
	int task_id = my_data->thread_id;
	int width = my_data->width;
	int i = my_data->i;
	double alpha = my_data->alpha;
	double* A = my_data->a;
	double* u = my_data->u;

	low = i + 1 + task_id * width;
	high = (task_id == NUM_THREADS-1)?(M-1):(low+width-1);

	for(j=low;j<high;j++)
	{
		aa=0.0;
		for(l=i;l<M;l++) 
			aa=aa+u[l]*A[l*M+j];
		aa=alpha*aa;
		for(l=i;l<M;l++) 
			A[l*M+j]=A[l*M+j]-aa*u[l];
	}

	pthread_exit(NULL);

}

/*used to create some threads*/
void pt_householder (double A[M][N],double y[M],int i)
{
	int j,l;
	static double aa,ki,u[M],alpha;
	aa=0.0;
	for (l=i;l<M;l++) 
		aa=aa+A[l][i]*A[l][i];
	aa=sqrt(aa);
	if (A[i][i]>0.0) 
		ki=-aa;
	else 
		ki=aa;
	alpha=1.0/(ki*(ki-A[i][i]));
	u[i]=A[i][i]-ki;
	A[i][i]=ki;
	for(l=i+1;l<M;l++) 
		{
			u[l]=A[l][i];
			A[l][i]=0.0;
		}
	int t;
	pthread_t threads[NUM_THREADS];
	struct thread_data thread_data_array[NUM_THREADS];
	int rc;
	int NUM;
	int width;

	NUM = ((N-i-1) <= NUM_THREADS)? (N-i-1):NUM_THREADS; 
	width = ((N-i-1) <= NUM_THREADS)? 1:(int)((N-i-1)/NUM_THREADS); 
	
	for (t = 0; t < NUM; t++)
	{
		thread_data_array[t].thread_id = t;
		thread_data_array[t].width = width;
		thread_data_array[t].i = i;
		thread_data_array[t].alpha = alpha;
		thread_data_array[t].a = A;
		thread_data_array[t].u = u;

		rc = pthread_create(&threads[t], NULL, work, (void*) &thread_data_array[t]);
		if (rc)
		{
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
	}

	for (t = 0; t < NUM; t++) {
		if (pthread_join(threads[t],NULL)){
			printf("ERROR; code on return from join is %d\n", rc);
			exit(-1);
		}
	}
	aa=0.0;
	for(l=i;l<M;l++) 
		aa=aa+u[l]*y[l];
	aa=alpha*aa;
	for(l=i;l<M;l++) 
		y[l]=y[l]-aa*u[l];
}


void pofitho (double y[M],double b[N])
{
	int i,j;
	double A[M][N];
	Input_1(A);
	for (i=0;i<N;i++)
		householder(A,y,i);	
	for (i=N-1;i>=0;i--)
	{
		for (j=i+1;j<N;j++) y[i]=y[i]-A[i][j]*b[j];
		b[i]=y[i]/A[i][i];
	}
}


void pt_pofitho (double y[M],double b[N])
{
	int i,j;
	double A[M][N];
	Input_1(A);
	for (i=0;i<N;i++)
		pt_householder(A,y,i);
	for (i=N-1;i>=0;i--)
	{
		for (j=i+1;j<N;j++) 
			y[i]=y[i]-A[i][j]*b[j];
		b[i]=y[i]/A[i][i];
	}
}

/*main function used to test time for the functions*/
main(int argc, char *argv[])
{
	int OPTION;
	struct timespec diff(struct timespec start, struct timespec end);
	struct timespec time1, time2;
	struct timespec time_stamp[OPTIONS];
	int clock_gettime(clockid_t clk_id, struct timespec *tp);
	long int j;
	long int time_sec, time_ns;



	OPTION = 0;
	int i;
	double b[N];
	double y[M];
	printf("Householder transformation:\n");
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
	pofitho(y,b);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
	time_stamp[OPTION] = diff(time1,time2);



	OPTION = 1;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
	pt_pofitho(y,b);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
	time_stamp[OPTION] = diff(time1,time2);



	for (j = 0; j < OPTIONS; j++) 
	{
	printf("time consumed is %ld ns\n", (long int)(
		 (GIG*time_stamp[j].tv_sec + time_stamp[j].tv_nsec)));
	}
	printf("\n");

}

/**********************************************/

struct timespec diff(struct timespec start, struct timespec end)
{
	struct timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) 
	{
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} 
	else
	{
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}

/*************************************************/
