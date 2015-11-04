#ifndef __HOUSEHOLDER_H__
#define __HOUSEHOLDER_H__


#include "matrix.h"

/* QR decomposition using Householder Reflections */
extern void qr_householder(const matrix *a, matrix *q, matrix *r);

#endif /* __HOUSEHOLDER_H__ */

