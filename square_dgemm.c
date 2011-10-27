#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "mkl.h"

#include "square_dgemm.h"

void square_dgemm(const int n, const double *A, const double *B, double *C) {

  	const char trans = 'N';
  	const double alpha = 1.0;
  	const double beta = 1.0;

	dgemm(&trans,&trans, &n, &n,&n, &alpha, A, &n, B, &n, &beta, C, &n);
}
