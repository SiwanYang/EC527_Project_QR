#ifndef __GIVENS_H__
#define __GIVENS_H__


#include "matrix.h"
/* QR decomposition using Givens Rotations */
extern void qr_givens(const matrix *a, matrix *q, matrix *r);
#endif /* __GIVENS_H__ */

