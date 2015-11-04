#include "householder.h"
#include "givens.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char **argv)
{
    unsigned int size;
    unsigned int row, col, count;
    clock_t start, finish;
    double elapsed_time;
    double h_time = 0.0, g_time = 0.0;
    matrix a, q, r;
    
    if (argc != 2)
    {
        /* display usage message and quit */
        (void)fprintf(stderr, "Usage: qr <matrix-size>\n\n");
        return EXIT_FAILURE;
    }
    /* convert the command line arguments to the appropriate parameters */
    size = atoi(argv[1]);
    /* initialize the random number generator */
    srand((unsigned)time(NULL));

    /* allocate memory for the matrices */
    matrix_malloc(&a, size, size);
    matrix_malloc(&q, size, size);
    matrix_malloc(&r, size, size);

    /* run the tests */    
    /* fill the a matrix with random numbers between -1 and 1 */
    for (row=0; row<size; row++)
    {
        for (col=0; col<size; col++)
        {
            MATRIX_GET_ELEMENT(&a, row, col) = ((double)rand() * 2.0 /
                RAND_MAX) - 1.0;
        }
    }


    /* run the Householder algorithm */
    (void)printf("  QR Decomposition using Householder Reflections\n");
    start = clock();
    qr_householder(&a, &q, &r);
    finish = clock();
    //calculate the time for Householder algorithm 
    elapsed_time = (double)(finish - start) / CLOCKS_PER_SEC;
    h_time += elapsed_time;
    (void)printf("    Time: %.6f sec.\n", elapsed_time);
        
    /* run the Givens algorithm */
    (void)printf("  QR Decomposition using Givens Rotations\n");
    start = clock();
    qr_givens(&a, &q, &r);
    finish = clock();	
    //calculate the time for Givens algorithm 
    elapsed_time = (double)(finish - start) / CLOCKS_PER_SEC;
    g_time += elapsed_time;
    (void)printf("    Time: %.6f sec.\n", elapsed_time);
	
    /* free memory used by matrices */
    matrix_free(&a);
    matrix_free(&q);
    matrix_free(&r);

    /* return success */
    return EXIT_SUCCESS;
}

