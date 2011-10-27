/**
 *  \file matmult.c
 *  \brief Matrix Multiply file for Proj1
 *  \author Aparna Chandramowlishwaran <aparna@gatech...>, Kent Czechowski <kentcz@gatech...>, Rich Vuduc <richie@gatech...>
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


void 
matmult (const int lda, const double *A, const double *B, double *C) 
{
  int i;

#pragma omp parallel for shared (lda,A,B,C) private (i)
  for (i = 0; i < lda; i++) {
    int j;
    for (j = 0; j < lda; j++) {
      double dotprod = 0.0; /* Accumulates the sum of the dot-product */
      int k;
      for (k = 0; k < lda; k++) {
        int a_index, b_index;
        a_index = (k * lda) + i; /* Compute index of A element */
        b_index = (j * lda) + k; /* Compute index of B element */
        dotprod += A[a_index] * B[b_index]; /* Compute product of A and B */
      } 
      int c_index = (j * lda) + i;
      C[c_index] = dotprod + C[c_index];
    } 
  } 
}
