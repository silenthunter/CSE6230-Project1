/**
 *  \file local_mm.c
 *  \brief Matrix Multiply file for Proj1
 *  \author Kent Czechowski <kentcz@gatech...>, Rich Vuduc <richie@gatech...>
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

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

#if defined(USE_TLB)
# include <string.h>
#endif

#if defined(USE_OPEN_MP)
# include <omp.h>
#endif

#ifdef USE_MKL
# include <mkl.h>
#endif

#if !defined(USE_MKL) && !defined(USE_OPEN_MP) && !defined(USE_TLB) && !defined(USE_BLOCKING)
# define USE_ORIGINAL
#endif

#define MIN(a, b)   ((a < b) ? a : b)
#define MAX(a, b)   ((a > b) ? a : b)

static void print_matrix(int rows, int cols, const double *mat) {

  int r, c;

  /* Iterate over the rows of the matrix */
  for (r = 0; r < rows; r++) {
    /* Iterate over the columns of the matrix */
    for (c = 0; c < cols; c++) {
      int index = (c * rows) + r;
      fprintf(stderr, "%.0lf ", mat[index]);
    } /* c */
    fprintf(stderr, "\n");
  } /* r */
}

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

  /* Verify the sizes of lda, ladb, and ldc */
  assert(lda >= m);
  assert(ldb >= k);
  assert(ldc >= m);

#ifdef USE_BLOCKING

#ifdef NONO
  int i, j;
  double* At; /* A transposed, aligned to L2 TLB page */

  /* Work on a portion of A */
  const int num_a_blocks = MAX(1, sizeof(double)*m*k / L2_CACHE_SIZE / 2);
  int a_block_idx;
  const int num_at_rows = m / num_a_blocks;

  assert(0 == posix_memalign((void**)&At, L2_TLB_PAGE_SIZE, L2_CACHE_SIZE));

  for (a_block_idx = 0; a_block_idx < num_a_blocks; a_block_idx++)
  {
    int a_row;
    int a_col;

    /* Transpose A and preload L2 cache */
    for (a_row = 0; a_row < m / num_a_blocks; a_row++)
    {
      for (a_col = 0; a_col < k; a_col++)
      {
        At[a_col + a_row*num_at_rows] = A[a_row + a_col*m];
        assert(a_col + a_row*num_at_rows < m*k);
      }
    }
#endif

  int k_block;
  int i_block;
  int j_block;
  double* CC;

  /* Allocate a buffer to accumulate the C result */
  assert(0 == posix_memalign((void**)&CC, L2_TLB_PAGE_SIZE, m*n*sizeof(double)));

  /* 8/16/32 utilizes 7168 bytes in the inner loop */
  /*
  const int bk = 8;
  const int bm = 16;
  const int bn = 32;
  */

  /* 4/8/8 utilizes 1024 bytes in the inner loop */
  /*
  const int bk = 4;
  const int bm = 8;
  const int bn = 8;
  */

  /* 32/32/32 utilizes 24K in the inner loop */
  const int bk = 32;
  const int bm = 32;
  const int bn = 32;

  /* I blocks increase top to bottom on A/C matrix */
  for (i_block = 0; i_block < m/bm; i_block++)
  {

    /* J blocks increase left to right on B/C matrix */
    for (j_block = 0; j_block < n/bn; j_block++)
    {
      int apply_beta = 1;

      /* K blocks increase top to bottom on B matrix (and left to right on A) */
      for (k_block = 0; k_block < k/bk; k_block++)
      {
        int block_row;
        int block_col;

        /* Iterate over the columns of C */
        for (block_col = 0; block_col < bn; block_col++) {

          /* Iterate over the rows of C */
          for (block_row = 0; block_row < bm; block_row++) {

            int k_iter;
            double dotprod = 0.0; /* Accumulates the sum of the dot-product */

            /* Iterate over column of A, row of B */
            for (k_iter = 0; k_iter < bk; k_iter++) {
              int a_index, b_index;
              a_index = k_iter*m + (i_block*bm) + (k_block*bk*m) + block_row;
              b_index = block_col*k + (j_block*bn*k) + (k_block*bk) + k_iter;
              dotprod += A[a_index] * B[b_index];
            } /* k_iter */

            int c_index = (block_col + j_block*bn) * m + (block_row + i_block*bm);
            if (apply_beta)
            {
              C[c_index] = alpha*dotprod + beta * C[c_index];
            }
            else
            {
              C[c_index] = alpha*dotprod + C[c_index];
            }
          } /* block_row */
        } /* block_col */
        apply_beta = 0;
      }
    }
  }

#endif /* USE_BLOCKING */

#ifdef USE_TLB
  int row, col;
  int i, j;
  double* b2 = (double*)malloc(sizeof(double) * k * n);
  memcpy(b2, A, sizeof(double) * k * n);

  //Transpose A
  for(i = 0; i < k; i++)
  {
    for(j = i + 1; j < n; j++)
    {
      double tmp = b2[i + j * m];
      b2[i + j * m] = b2[j + i * m];
      b2[j + i * m] = tmp;
    }
  }

  /* Iterate over the columns of C */
  for (col = 0; col < n; col++) {

    /* Iterate over the rows of C */
    for (row = 0; row < m; row++) {

      int k_iter;
      double dotprod = 0.0; /* Accumulates the sum of the dot-product */

      /* Iterate over column of A, row of B */
      for (k_iter = 0; k_iter < k; k_iter++) {
        int a_index, b_index;
        a_index = (row * lda) + k_iter; /* Compute index of A element */
        b_index = (col * ldb) + k_iter; /* Compute index of B element */
        dotprod += b2[a_index] * B[b_index]; /* Compute product of A and B */
      } /* k_iter */

      int c_index = (col * ldc) + row;
      C[c_index] = (alpha * dotprod) + (beta * C[c_index]);
    } /* row */
  } /* col */

  free(b2);
#endif /* USE_TLB */

#ifdef USE_OPEN_MP
  int row, col;

# pragma omp parallel for private(col, row)
  /* Iterate over the columns of C */
  for (col = 0; col < n; col++) {

    /* Iterate over the rows of C */
    for (row = 0; row < m; row++) {

      int k_iter;
      double dotprod = 0.0; /* Accumulates the sum of the dot-product */

      /* Iterate over column of A, row of B */
      for (k_iter = 0; k_iter < k; k_iter++) {
        int a_index, b_index;
        a_index = (k_iter * lda) + row; /* Compute index of A element */
        b_index = (col * ldb) + k_iter; /* Compute index of B element */
        dotprod += A[a_index] * B[b_index]; /* Compute product of A and B */
      } /* k_iter */

      int c_index = (col * ldc) + row;
      C[c_index] = (alpha * dotprod) + (beta * C[c_index]);
    } /* row */
  } /* col */
#endif /* USE_OPEN_MP */

#ifdef USE_MKL
  const char N = 'N';
  dgemm(&N, &N, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
#endif

#ifdef USE_ORIGINAL
  int row, col;

  /* Iterate over the columns of C */
  for (col = 0; col < n; col++) {

    /* Iterate over the rows of C */
    for (row = 0; row < m; row++) {

      int k_iter;
      double dotprod = 0.0; /* Accumulates the sum of the dot-product */

      /* Iterate over column of A, row of B */
      for (k_iter = 0; k_iter < k; k_iter++) {
        int a_index, b_index;
        a_index = (k_iter * lda) + row; /* Compute index of A element */
        b_index = (col * ldb) + k_iter; /* Compute index of B element */
        dotprod += A[a_index] * B[b_index]; /* Compute product of A and B */
      } /* k_iter */

      int c_index = (col * ldc) + row;
      C[c_index] = (alpha * dotprod) + (beta * C[c_index]);
    } /* row */
  } /* col */
#endif

}
