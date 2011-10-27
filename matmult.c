/**
 *  \file matmult.c
 *  \brief Matrix Multiply file for Proj1
 *  \author Kent Czechowski <kentcz@gatech...>, Rich Vuduc <richie@gatech...>
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "matmult.h"

/**
 *
 *  Matrix Multiply
 *   Computes C = A * B + C
 *
 *
 *  Similar to the DGEMM routine in BLAS
 *
 *
 **/

void matmult(const int n, const double *A, const double *B, double *C) {
  int row, col;

  /* Iterate over the columns of C */
  for (col = 0; col < n; col++) {

    /* Iterate over the rows of C */
    for (row = 0; row < n; row++) {

      int k_iter;
      double dotprod = 0.0; /* Accumulates the sum of the dot-product */

      /* Iterate over column of A, row of B */
      for (k_iter = 0; k_iter < n; k_iter++) {
        int a_index, b_index;
        a_index = (k_iter * n) + row; /* Compute index of A element */
        b_index = (col * n) + k_iter; /* Compute index of B element */
        dotprod += A[a_index] * B[b_index]; /* Compute product of A and B */
      } /* k_iter */

      int c_index = (col * n) + row;
      C[c_index] = dotprod + C[c_index];
    } /* row */
  } /* col */

}
