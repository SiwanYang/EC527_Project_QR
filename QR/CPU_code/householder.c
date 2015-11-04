#include "matrix.h"
/* special subtraction routine used only by Householder reflections;
   subtracts two matrics of different sizes, the smaller being aligned with
   the bottom-right corner of the larger matrix; stores in left_large */
static void householder_corner_subtraction(matrix *left_large,
                                           const matrix *right_small)
{
    unsigned int row, col;
    unsigned int row_offset, col_offset;

#ifdef __CHECK_DIMENSIONS__
    /* check the dimensions of the result matrices */
    if ((left_large->rows < right_small->rows) ||
        (left_large->cols < right_small->cols))
    {
        error("Invalid matrix dimensions in matrix_corner_subtraction()!");
        return;
    }
#endif /* __CHECK_DIMENSIONS__ */

    /* calculate start offsets for the large matrix */
    row_offset = left_large->rows - right_small->rows;
    col_offset = left_large->cols - right_small->cols;

    /* subtract the elements individually */
    for (row=0; row<right_small->rows; row++)
    {
        for (col=0; col<right_small->cols; col++)
        {
            MATRIX_GET_ELEMENT(left_large, row_offset + row, col_offset +
                col) -= MATRIX_GET_ELEMENT(right_small, row, col);
        }
    }
}

extern void qr_householder(const matrix *a, matrix *q, matrix *r)
{
    matrix q_copy, r_copy;
    matrix x, h, h_temp;
    unsigned int col;

#ifdef __CHECK_DIMENSIONS__
    /* check the dimensions of the result matrices */
    if ((a->rows != a->cols) || (q->rows != q->cols) ||
        (r->rows != r->cols) || (a->rows != q->rows) || (a->rows != r->rows))
    {
        error("Invalid matrix dimensions in qr_householder()!");
        return;
    }
#endif /* __CHECK_DIMENSIONS__ */

    /* initialize the matrices */
    matrix_malloc(&q_copy, q->rows, q->cols);
    matrix_malloc(&r_copy, r->rows, r->cols);
    matrix_malloc(&h, a->rows, a->cols);
    matrix_empty(&x);
    matrix_empty(&h_temp);
    matrix_identity(q);
    matrix_copy(a, r);

    /* find each of the Householder reflection matrices */
    for (col=0; col<a->cols-1; col++)
    {
        /* get the x vector */
        matrix_realloc(&x, a->cols - col, 1);
        matrix_get_col_vector(r, col, col, r->cols - col, &x);

        /* make x into u~ (it only involves changing one element, so there's
           no point in allocating memory for a brand new vector) */
        if (MATRIX_GET_ELEMENT(&x, 0, 0) >= 0)
            MATRIX_GET_ELEMENT(&x, 0, 0) += vector_col_norm(&x);
        else
            MATRIX_GET_ELEMENT(&x, 0, 0) -= vector_col_norm(&x);

        /* create h_temp (the 2uu^t part of h) */
        matrix_realloc(&h_temp, x.rows, x.rows);
        matrix_col_vector_transpose_multiplication(&x, &h_temp);
        matrix_constant_multiplication(&h_temp, (matrix_element)2 /
            vector_col_dot_col(&x, &x), &h_temp);

        /* create h (I - 2uu^t) */
        matrix_identity(&h);
        householder_corner_subtraction(&h, &h_temp);

        /* "add" h to the q and r matrices and this part is the main optimization for the cpu code*/
        matrix_copy(q, &q_copy);
        matrix_copy(r, &r_copy);
        matrix_multiplication(&q_copy, &h, q);
        matrix_multiplication(&h, &r_copy, r);
    }

    /* free memory */
    matrix_free(&q_copy);
    matrix_free(&r_copy);
    matrix_free(&h);
    matrix_free(&x);
    matrix_free(&h_temp);
}

