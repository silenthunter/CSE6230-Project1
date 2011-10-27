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

#if !defined (BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif

#define MIN(a, b) (a < b) ? a : b

void 
basic_dgemm (const int lda, const int M, const int N, const int K, const double *A, const double *B, double *C)
{
  int i, j, k;

  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      double dotprod = 0.0; /* Accumulates the sum of the dot-product */
      for (k = 0; k < K; k++) {
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

void 
matmult (const int lda, const double *A, const double *B, double *C) 
{
  int i;

#pragma omp parallel for shared (lda,A,B,C) private (i)
  for (i = 0; i < lda; i += BLOCK_SIZE) {
    int j;
    for (j = 0; j < lda; j += BLOCK_SIZE) {
      int k;
      for (k = 0; k < lda; k += BLOCK_SIZE) {
        int M = MIN (BLOCK_SIZE, lda-i);
        int N = MIN (BLOCK_SIZE, lda-j);
        int K = MIN (BLOCK_SIZE, lda-k);

        basic_dgemm (lda, M, N, K, A+i+k*lda, B+k+j*lda, C+i+j*lda);
      }
    }
  }  

}
