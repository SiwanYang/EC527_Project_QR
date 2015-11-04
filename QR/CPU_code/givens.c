#include "matrix.h"
#include <math.h>


extern void qr_givens(const matrix *a, matrix *q, matrix *r)
{
    matrix q_copy, r_copy, R, Rt;
    unsigned int row, col;
    double x, y, xydist, negsine, cosine;

#ifdef __CHECK_DIMENSIONS__
    /* check the dimensions of the result matrices */
    if ((a->rows != a->cols) || (q->rows != q->cols) ||
        (r->rows != r->cols) || (a->rows != q->rows) || (a->rows != r->rows))
    {
        error("Invalid matrix dimensions in qr_givens()!");
        return;
    }
#endif /* __CHECK_DIMENSIONS__ */

    /* initialize the matrices */
    matrix_malloc(&q_copy, q->rows, q->cols);
    matrix_malloc(&r_copy, r->rows, r->cols);
    matrix_malloc(&R, a->rows, a->cols);
    matrix_malloc(&Rt, a->cols, a->rows);
    matrix_identity(q);
    matrix_copy(a, r);

    /* find each of the Givens rotation matrices */
    for (col=0; col<a->cols-1; col++)
    {
        for (row=col+1; row<a->rows; row++)
        {
            /* only compute a rotation matrix if the corresponding entry in
               the matrix is nonzero */
            y = MATRIX_GET_ELEMENT(r, row, col);
            if (y != 0.0)
            {
                /* find the values of -sin and cos */
                x = MATRIX_GET_ELEMENT(r, col, col);
                xydist = sqrt(x * x + y * y);
                negsine = y / xydist;
                cosine = x / xydist;
                
                /* calculate the R and R^t matrices */
                matrix_identity(&R);
                MATRIX_GET_ELEMENT(&R, col, col) = cosine;
                MATRIX_GET_ELEMENT(&R, col, row) = negsine;
                MATRIX_GET_ELEMENT(&R, row, col) = -negsine;
                MATRIX_GET_ELEMENT(&R, row, row) = cosine;
                matrix_transpose(&R, &Rt);

                /* "add" R to the q and r matrices */
                matrix_copy(q, &q_copy);
                matrix_copy(r, &r_copy);
		int i;
		for(i=0;i<q->rows;i++)
			{
				givens_multipulication(&q_copy,&Rt,q,row,i);
				givens_multipulication(&q_copy,&Rt,q,col,i);			
				givens_multipulication(&R, &r_copy, r,i,row);
				givens_multipulication(&R, &r_copy, r,i,col);
			}
            }
        }
    }

    /* free memory */
    matrix_free(&q_copy);
    matrix_free(&r_copy);
    matrix_free(&R);
    matrix_free(&Rt);
}

