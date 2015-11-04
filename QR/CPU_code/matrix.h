#ifndef __MATRIX_H__
#define __MATRIX_H__




/* the precision of the matrix operations can be modified by changing the
   following typedef */
typedef double matrix_element;


/* the matrix structure */
typedef struct
{
    matrix_element *elements;
    unsigned int rows, cols;
} matrix;


/* memory management functions */
extern void matrix_empty(matrix *ptr);
extern void matrix_malloc(matrix *ptr, const unsigned int rows,
                          const unsigned int cols);
extern void matrix_realloc(matrix *ptr, const unsigned int rows,
                           const unsigned int cols);
extern void matrix_free(matrix *ptr);


/* matrix element accessor */
#define MATRIX_GET_ELEMENT(M, row, col) \
    ((M)->elements[((col) * (M)->rows) + (row)])


/* common matrix operations */
extern void matrix_identity(matrix *result);

extern void matrix_addition(const matrix *left, const matrix *right,
                            matrix *result);
extern void matrix_subtraction(const matrix *left, const matrix *right,
                               matrix *result);

extern void matrix_multiplication(const matrix *left, const matrix *right,
                                  matrix *result);

extern void matrix_constant_multiplication(const matrix *original,
                                           const matrix_element constant,
                                           matrix *result);
extern void matrix_constant_division(const matrix *original,
                                     const matrix_element constant,
                                     matrix *result);

extern matrix_element vector_dot_product(const matrix *vector1,
                                         const matrix *vector2);
extern matrix_element vector_col_dot_col(const matrix *vector1,
                                         const matrix *vector2);
extern matrix_element vector_col_dot_row(const matrix *vector1,
                                         const matrix *vector2);
extern matrix_element vector_row_dot_col(const matrix *vector1,
                                         const matrix *vector2);
extern matrix_element vector_row_dot_row(const matrix *vector1,
                                         const matrix *vector2);

extern matrix_element vector_norm(const matrix *vector);
extern matrix_element vector_col_norm(const matrix *vector);
extern matrix_element vector_row_norm(const matrix *vector);

extern void matrix_copy(const matrix *src, matrix *dest);
extern void matrix_transpose(const matrix *src, matrix *dest);


/* some less-common, but useful matrix operations */
extern void matrix_get_col_vector(const matrix *source,
                                  const unsigned int col,
                                  const unsigned int row_start,
                                  const unsigned int elements,
                                  matrix *vector);
extern void matrix_get_row_vector(const matrix *source,
                                  const unsigned int row,
                                  const unsigned int col_start,
                                  const unsigned int elements,
                                  matrix *vector);

extern void matrix_col_vector_transpose_multiplication(const matrix *vector,
                                                       matrix *result);
extern void matrix_row_vector_transpose_multiplication(const matrix *vector,
                                                       matrix *result);

extern double matrix_get_max_element(const matrix *src);


#ifdef __CHECK_DIMENSIONS__
/* error handling */
extern void error(const char *msg);
#endif /* __CHECK_DIMENSIONS__ */


#endif /* __MATRIX_H__ */

extern void givens_multipulication(const matrix *left, const matrix *right,
                                  matrix *result,int col,int row);

