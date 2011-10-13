/**
 *  \file local_mm.c
 *  \brief Matrix Multiply file for Proj1
 *  \author Kent Czechowski <kentcz@gatech...>, Rich Vuduc <richie@gatech...>
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include "matrix_utils.h"

#define MAX(a, b) ((a < b) ? b : a)
#define MIN(a, b) ((a < b) ? a : b)

#define MB *1024*1024
#define KB *1024

/* Intel Xeon X5650 CPU Gulftown (sixcore) */
/* L1 and L2 cache are per core */
#define L1_DATA_CACHE_SIZE (32 KB)
#define L2_CACHE_SIZE (256 KB)
#define L3_CACHE_SIZE (12 MB)

#define TLB0_PAGE_SIZE (2 MB)
#define TLB0_PAGE_SIZE_MAYBE (4 MB)
#define TLB0_PAGE_ENTRIES 32
#define L2_TLB_PAGE_ENTRIES 512

#define TLB_PAGE_SIZE (4 KB)
#define TLB_PAGE_ENTRIES 64
#define L2_TLB_PAGE_SIZE (4 KB)

/**
 *
 *  Local Matrix Multiply
 *   Computes C = alpha * A * B + beta * C
 *
 *
 *  Similar to the DGEMM routine in BLAS
 *
 *
 *  alpha and beta are double-precision scalars
 *
 *  A, B, and C are matrices of double-precision elements
 *  stored in column-major format 
 *
 *  The output is stored in C
 *  A and B are not modified during computation
 *
 *
 *  m - number of rows of matrix A and rows of C
 *  n - number of columns of matrix B and columns of C
 *  k - number of columns of matrix A and rows of B
 * 
 *  lda, ldb, and ldc specifies the size of the first dimension of the matrices
 *
 **/
void local_mm(const int m, const int n, const int k, const double alpha,
    const double *A, const int lda, const double *B, const int ldb,
    const double beta, double *C, const int ldc) {

  int row, col, i, j, proc;

  /* Verify the sizes of lda, ladb, and ldc */
  assert(lda >= m);
  assert(ldb >= k);
  assert(ldc >= m);

  //print_matrix(m, k, A);


  for(proc = 0; proc < 2; proc++)//omp_get_num_threads(); proc++)
  {
  printf("Loop\n");
  int threads = 2;//omp_get_num_threads();
  int width = m / threads;

  double* b2 = (double*)malloc(sizeof(double) * width * k);

  //print_matrix(m, k, A);

  //Copy transpose A
  for(i = 0; i < width * k; i++)
  {
    b2[i] = A[(i % k) * m + i / k + proc * width];
    //printf("idx: %d, A: %f\n", (i % k) * m + i / k + proc * width, b2[i]);
  }

  /*for(i = 0; i < k * width; i++)
  {
    printf("%f ", b2[i]);
    if(i % k == k - 1) printf("\n");
  }*/

  /* Iterate over the columns of C */
  for (col = 0; col < n; col++) {

    /* Iterate over the rows of C */
    for (row = proc * width; row < proc * width + width; row++) {

      int k_iter;
      double dotprod = 0.0; /* Accumulates the sum of the dot-product */

      /* Iterate over column of A, row of B */
      for (k_iter = 0; k_iter < k; k_iter++) {
        int a_index, b_index;
        a_index = (row * k) + k_iter - proc * width * k; /* Compute index of A element */
        b_index = (col * ldb) + k_iter; /* Compute index of B element */
        dotprod += b2[a_index] * B[b_index]; /* Compute product of A and B */
        //printf("C: %d, A: %d, B: %d, prod: %f\n", col * ldc + row, a_index, b_index, dotprod);
      } /* k_iter */

      int c_index = (col * ldc) + row;
      C[c_index] = (alpha * dotprod) + (beta * C[c_index]);
    } /* row */
  } /* col */

  free(b2);
}

}
