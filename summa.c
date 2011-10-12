/**
 *  \file summa.c
 *  \brief Implementation of Scalable Universal
 *    Matrix Multiplication Algorithm for Proj1
 */

//#define PDEBUG

#define NDEBUG

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <unistd.h>

#include "local_mm.h"

//#define PDEBUG

/* My debug print macro */
#ifdef PDEBUG
#define RANK_PRINTF(__rank__, __format__, args...) \
  if (rank == __rank__) \
  { \
    fprintf(stderr, __format__, ## args); \
  }

#define RANK_FN(__rank__, fn, args...) \
  if (rank == __rank__) \
  { \
    (*fn)(args); \
  }
#else
#define RANK_PRINTF(args...)
#define RANK_FN(args...)
#endif

static void print_matrix_int(int rows, int cols, int *mat) {

  int r, c;

  /* Iterate over the rows of the matrix */
  for (r = 0; r < rows; r++) {
    /* Iterate over the columns of the matrix */
    for (c = 0; c < cols; c++) {
      int index = (c * rows) + r;
      fprintf(stderr, "%2i ", mat[index]);
    } /* c */
    fprintf(stderr, "\n");
  } /* r */
}

static void print_matrix(int rows, int cols, double *mat) {

  int r, c;

  /* Iterate over the rows of the matrix */
  for (r = 0; r < rows; r++) {
    /* Iterate over the columns of the matrix */
    for (c = 0; c < cols; c++) {
      int index = (c * rows) + r;
      fprintf(stderr, "%2.0lf ", mat[index]);
    } /* c */
    fprintf(stderr, "\n");
  } /* r */
}

/* For A matrix, set panel_ranks with the horizontal rank assignments */
static void set_horizontal_ranks(int rank, int procGridX, int procGridY, int* panel_ranks)
{
  int i;
  int row;
  int col;

  row = rank % procGridX;
  col = (rank - row) / procGridX;

  assert(row < procGridX);
  assert(col < procGridY);

  for (i = 0; i < procGridY; i++)
  {
    panel_ranks[i] = row + i*procGridX;
    assert(i < procGridY);
  }
}

/* For B matrix, set panel_ranks with the vertical rank assignments */
static void set_vertical_ranks(int rank, int procGridX, int procGridY, int* panel_ranks)
{
  int i;
  int row;
  int col;

  row = rank % procGridX;
  col = (rank - row) / procGridX;

  assert(row < procGridX);
  assert(col < procGridY);

  for (i = 0; i < procGridX; i++)
  {
    panel_ranks[i] = col*procGridX + i;
    assert(i < procGridX);
  }
}

/**
 * Distributed Matrix Multiply using the SUMMA algorithm
 *  Computes C = A*B + C
 *
 *  This function uses procGridX times procGridY processes
 *   to compute the product
 *
 *  A is a m by k matrix, each process starts
 *	with a block of A (aBlock)
 *
 *  B is a k by n matrix, each process starts
 *	with a block of B (bBlock)
 *
 *  C is a m by n matrix, each process starts
 *	with a block of C (cBlock)
 *
 *  The resulting matrix is stored in C.
 *  A and B should not be modified during computation.
 *
 *  Ablock, Bblock, and CBlock are stored in
 *   column-major format
 *
 *  pb is the Panel Block Size
 **/
static void summa_small(int m, int n, int k, double *Ablock, double *Bblock, double *Cblock,
		int procGridX, int procGridY, int pb)
{
  int rank;
  double* Apanel;
  double* Bpanel;
  int* vertical_neighbors;
  int* horizontal_neighbors;

  MPI_Group orig_group;
  MPI_Group vertical_group;
  MPI_Group horizontal_group;
  MPI_Comm vertical_comm;
  MPI_Comm horizontal_comm;
  int vertical_group_rank;
  int horizontal_group_rank;

  int a_block_num_columns;
  int a_block_num_rows;
  int b_block_num_columns;
  int b_block_num_rows;

  int pb_k;
  int r;
  int c;
  int xx = 5;

  a_block_num_columns = k/procGridY;
  a_block_num_rows = m/procGridX;
  b_block_num_columns = n/procGridY;
  b_block_num_rows = k/procGridX;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Apanel = calloc(1, sizeof(double) * pb * a_block_num_rows);
  assert(Apanel);

  Bpanel = calloc(1, sizeof(double) * pb * b_block_num_columns);
  assert(Bpanel);

  vertical_neighbors = malloc(sizeof(int) * procGridX);
  assert(vertical_neighbors);

  horizontal_neighbors = malloc(sizeof(int) * procGridY);
  assert(horizontal_neighbors);

  RANK_PRINTF(xx, "*\n*\n*\n*\n*\n*\n*\n*\n*\n*\nRank %i\n", rank);
  RANK_PRINTF(xx, "m=%d, n=%d, k=%d, procGridX=%d, procGridY=%d, pb=%i\n",
              m, n, k, procGridX, procGridY, pb);

  /* Configure ranks for A's horizontal panel Bcast containing the current rank */
  set_horizontal_ranks(rank, procGridX, procGridY, horizontal_neighbors);
  RANK_PRINTF(xx, "horizontal_neighbors:\n");
  RANK_FN(xx, print_matrix_int, 1, procGridY, horizontal_neighbors);

  /* Configure ranks for B's vertical panel Bcast containing the current rank */
  set_vertical_ranks(rank, procGridX, procGridY, vertical_neighbors);
  RANK_PRINTF(xx, "vertical_neighbors:\n");
  RANK_FN(xx, print_matrix_int, 1, procGridX, vertical_neighbors);

  // Print each matrix (block)
  RANK_PRINTF(xx, "Ablock:\n");
  RANK_FN(xx, print_matrix, a_block_num_rows, a_block_num_columns, Ablock);

  RANK_PRINTF(xx, "Bblock:\n");
  RANK_FN(xx, print_matrix, b_block_num_rows, b_block_num_columns, Bblock);

  RANK_PRINTF(xx, "a_block_num_columns: %i\n", a_block_num_columns);
  RANK_PRINTF(xx, "b_block_num_rows: %i\n", b_block_num_rows);

  /* Create communicator groups */

  MPI_Comm_group(MPI_COMM_WORLD, &orig_group);
  MPI_Comm_group(MPI_COMM_WORLD, &vertical_group);
  MPI_Comm_group(MPI_COMM_WORLD, &horizontal_group);

  MPI_Group_incl(orig_group, procGridX, vertical_neighbors, &vertical_group);
  MPI_Comm_create(MPI_COMM_WORLD, vertical_group, &vertical_comm);
  MPI_Group_rank(vertical_group, &vertical_group_rank);

  MPI_Group_incl(orig_group, procGridY, horizontal_neighbors, &horizontal_group);
  MPI_Comm_create(MPI_COMM_WORLD, horizontal_group, &horizontal_comm);
  MPI_Group_rank(horizontal_group, &horizontal_group_rank);

  for (pb_k = 0; pb_k < k/pb; pb_k++)
  {
    int a_vertical_panel_src_rank_index;
    int b_horizontal_panal_src_rank_index;
    int a_vertical_panel_src_rank;
    int b_horizontal_panal_src_rank;

    RANK_PRINTF(xx, "***************************************************************\n");
    RANK_PRINTF(xx, "pb_k=%i\n", pb_k);

    /* Determine the rank of where required panels come from.  Note the confusing variable */
    /* names.  Variable a_vertical_panel_src_rank_index is an index into the */
    /* horizontal_neighbors array. */
    a_vertical_panel_src_rank_index = pb * pb_k / a_block_num_columns;
    b_horizontal_panal_src_rank_index = pb * pb_k / b_block_num_rows;

    assert(a_vertical_panel_src_rank_index < procGridY);
    assert(b_horizontal_panal_src_rank_index < procGridX);

    a_vertical_panel_src_rank = horizontal_neighbors[a_vertical_panel_src_rank_index];
    b_horizontal_panal_src_rank = vertical_neighbors[b_horizontal_panal_src_rank_index];

    RANK_PRINTF(xx, "a_vertical_panel_src_rank_index = %i\n",    a_vertical_panel_src_rank_index);
    RANK_PRINTF(xx, "b_horizontal_panal_src_rank_index = %i\n",  b_horizontal_panal_src_rank_index);
    RANK_PRINTF(xx, "a_vertical_panel_src_rank = %i\n",          a_vertical_panel_src_rank);
    RANK_PRINTF(xx, "b_horizontal_panal_src_rank = %i\n",        b_horizontal_panal_src_rank);
    RANK_PRINTF(xx, "pb_k=%i, k=%i, m=%i, procGridX=%i, procGridY=%i, pb=%i\n", pb_k, k, m, procGridX, procGridY, pb);

#ifdef PDEBUG
    fflush(stderr);
    MPI_Barrier(MPI_COMM_WORLD);
    fflush(stderr);
#endif

    assert(a_vertical_panel_src_rank < procGridY);
    assert(b_horizontal_panal_src_rank < procGridX);

    if (rank == a_vertical_panel_src_rank)
    {
      int panels_per_block = a_block_num_columns / pb;
      int panel_start_column = (pb_k % panels_per_block) * pb;

      /* I'm the horizontal neighbor source */
      RANK_PRINTF(xx, "I am the source of the Apanel\n");
      RANK_PRINTF(xx, "Sourcing starting from column %i of my Apanel\n", panel_start_column);
      RANK_PRINTF(xx, "panels_per_block = %i\n", panels_per_block);
      RANK_PRINTF(xx, "Ablock (source)\n");
      RANK_FN(xx, print_matrix, a_block_num_rows, a_block_num_columns, Ablock);

      /* Populate Apanel; a_block_num_rows by pb matrix */
#ifdef NOTGOOD
      for (r = 0; r < a_block_num_rows; r++)
      {
        for (c = 0; c < pb; c++)
        {
#endif
      for (c = 0; c < pb; c++)
      {
        for (r = 0; r < a_block_num_rows; r++)
        {
          int ablock_index = r + panel_start_column*a_block_num_rows + c*a_block_num_rows;

          Apanel[r + c*a_block_num_rows] = Ablock[ablock_index];

          assert(r + c*a_block_num_rows < pb * a_block_num_rows);
          assert(ablock_index < a_block_num_columns * a_block_num_rows);
        }
      }
    }

    if (rank == b_horizontal_panal_src_rank)
    {
      int panels_per_block = b_block_num_rows / pb;
      int panel_start_row = (pb_k % panels_per_block) * pb;

      /* I'm the vertical neighbor source */
      RANK_PRINTF(xx, "I am the source of the Bpanel\n");

      RANK_PRINTF(xx, "Sourcing starting from row %i of my Bpanel\n", panel_start_row);
      RANK_PRINTF(xx, "panels_per_block = %i\n", panels_per_block);
      RANK_PRINTF(xx, "Bblock (source)\n");
      RANK_FN(xx, print_matrix, b_block_num_rows, b_block_num_columns, Bblock);

      /* Populate Bpanel; pb by b_block_num_columns matrix */
#ifdef NOTGOOD
      for (r = 0; r < pb; r++)
      {
        for (c = 0; c < b_block_num_columns; c++)
        {
#endif
      for (c = 0; c < b_block_num_columns; c++)
      {
        for (r = 0; r < pb; r++)
        {
          int bblock_index = panel_start_row + r + c*b_block_num_rows;

          Bpanel[r + c*pb] = Bblock[bblock_index];

          assert(r + c*pb < pb * b_block_num_columns);
          assert(bblock_index < b_block_num_columns * b_block_num_rows);
        }
      }
    }

    RANK_PRINTF(xx, "Apanel (before potential transmit)\n");
    RANK_FN(xx, print_matrix, a_block_num_rows, pb, Apanel);

    RANK_PRINTF(xx, "Bpanel (before potential transmit)\n");
    RANK_FN(xx, print_matrix, pb, b_block_num_columns, Bpanel);

    MPI_Bcast(Apanel, pb * a_block_num_rows, MPI_DOUBLE, a_vertical_panel_src_rank_index, horizontal_comm);
    MPI_Bcast(Bpanel, pb * b_block_num_columns, MPI_DOUBLE, b_horizontal_panal_src_rank_index, vertical_comm);

    if (rank == xx)
    {
      if (rank != a_vertical_panel_src_rank)
      {
        RANK_PRINTF(xx, "Apanel **received** (%i <-- %i [index %i])\n", xx, a_vertical_panel_src_rank,
            a_vertical_panel_src_rank_index);
        RANK_FN(xx, print_matrix, a_block_num_rows, pb, Apanel);
      }
      if (rank != b_horizontal_panal_src_rank)
      {
        RANK_PRINTF(xx, "Bpanel **received** (%i <-- %i [index %i])\n", xx, b_horizontal_panal_src_rank,
            b_horizontal_panal_src_rank_index);
        RANK_FN(xx, print_matrix, pb, b_block_num_columns, Bpanel);
      }
    }

#ifdef PDEBUG
    fflush(stderr);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    local_mm(a_block_num_rows, b_block_num_columns, pb, 1.0,
        Apanel, a_block_num_rows, Bpanel, pb,
        1.0, Cblock, a_block_num_rows);
  }

  /* Free up allocated resources */
  MPI_Comm_free(&vertical_comm);
  MPI_Group_free(&vertical_group);

  MPI_Comm_free(&horizontal_comm);
  MPI_Group_free(&horizontal_group);

  free(horizontal_neighbors);
  free(vertical_neighbors);
  free(Bpanel);
  free(Apanel);

	MPI_Barrier(MPI_COMM_WORLD);
}

static void summa_large(int m, int n, int k, double *Ablock, double *Bblock, double *Cblock,
		int procGridX, int procGridY, int pb)
{
  int rank;
  double* Apanel;
  double* Bpanel;
  double* Asubpanel;
  double* Bsubpanel;
  int* vertical_neighbors;
  int* horizontal_neighbors;

  MPI_Group orig_group;
  MPI_Group vertical_group;
  MPI_Group horizontal_group;
  MPI_Comm vertical_comm;
  MPI_Comm horizontal_comm;
  int vertical_group_rank;
  int horizontal_group_rank;

  int a_block_num_columns;
  int a_block_num_rows;
  int b_block_num_columns;
  int b_block_num_rows;

  int pb_k;
  int r;
  int c;
  int xx = 5;

  a_block_num_columns = k/procGridY;
  a_block_num_rows = m/procGridX;
  b_block_num_columns = n/procGridY;
  b_block_num_rows = k/procGridX;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Asubpanel = calloc(1, sizeof(double) * pb * a_block_num_rows * a_block_num_columns);
  assert(Asubpanel);

  Apanel = calloc(1, sizeof(double) * pb * a_block_num_rows);
  assert(Apanel);

  Bsubpanel = calloc(1, sizeof(double) * pb * b_block_num_columns);
  assert(Bsubpanel);

  Bpanel = calloc(1, sizeof(double) * pb * b_block_num_columns * b_block_num_rows);
  assert(Bpanel);

  vertical_neighbors = malloc(sizeof(int) * procGridX);
  assert(vertical_neighbors);

  horizontal_neighbors = malloc(sizeof(int) * procGridY);
  assert(horizontal_neighbors);

  RANK_PRINTF(xx, "*\n*\n*\n*\n*\n*\n*\n*\n*\n*\nRank %i\n", rank);
  RANK_PRINTF(xx, "m=%d, n=%d, k=%d, procGridX=%d, procGridY=%d, pb=%i\n",
              m, n, k, procGridX, procGridY, pb);

  /* Configure ranks for A's horizontal panel Bcast containing the current rank */
  set_horizontal_ranks(rank, procGridX, procGridY, horizontal_neighbors);
  RANK_PRINTF(xx, "horizontal_neighbors:\n");
  RANK_FN(xx, print_matrix_int, 1, procGridY, horizontal_neighbors);

  /* Configure ranks for B's vertical panel Bcast containing the current rank */
  set_vertical_ranks(rank, procGridX, procGridY, vertical_neighbors);
  RANK_PRINTF(xx, "vertical_neighbors:\n");
  RANK_FN(xx, print_matrix_int, 1, procGridX, vertical_neighbors);

  // Print each matrix (block)
  RANK_PRINTF(xx, "Ablock:\n");
  RANK_FN(xx, print_matrix, a_block_num_rows, a_block_num_columns, Ablock);

  RANK_PRINTF(xx, "Bblock:\n");
  RANK_FN(xx, print_matrix, b_block_num_rows, b_block_num_columns, Bblock);

  /* Create communicator groups */

  MPI_Comm_group(MPI_COMM_WORLD, &orig_group);
  MPI_Comm_group(MPI_COMM_WORLD, &vertical_group);
  MPI_Comm_group(MPI_COMM_WORLD, &horizontal_group);

  MPI_Group_incl(orig_group, procGridX, vertical_neighbors, &vertical_group);
  MPI_Comm_create(MPI_COMM_WORLD, vertical_group, &vertical_comm);
  MPI_Group_rank(vertical_group, &vertical_group_rank);

  MPI_Group_incl(orig_group, procGridY, horizontal_neighbors, &horizontal_group);
  MPI_Comm_create(MPI_COMM_WORLD, horizontal_group, &horizontal_comm);
  MPI_Group_rank(horizontal_group, &horizontal_group_rank);

  for (pb_k = 0; pb_k < k; pb_k = pb_k + 1)
  {
    int a_vertical_subpanel_src_rank_index;
    int b_horizontal_subpanel_src_rank_index;
    int a_vertical_subpanel_src_rank;
    int b_horizontal_subpanel_src_rank;
    int a_subpanel_transmit_ready;
    int b_subpanel_transmit_ready;
    int subpanel_start_column = pb_k % a_block_num_columns;

    RANK_PRINTF(xx, "***************************************************************\n");
    RANK_PRINTF(xx, "pb_k=%i\n", pb_k);

    /* Determine the rank of where required panels come from.  Note the confusing variable */
    /* names.  Variable a_vertical_panel_src_rank_index is an index into the */
    /* horizontal_neighbors array. */
    a_vertical_subpanel_src_rank_index = pb_k / a_block_num_columns;
    b_horizontal_subpanel_src_rank_index = pb_k / b_block_num_rows;

    assert(a_vertical_subpanel_src_rank_index < procGridY);
    assert(b_horizontal_subpanel_src_rank_index < procGridX);

    a_vertical_subpanel_src_rank = horizontal_neighbors[a_vertical_subpanel_src_rank_index];
    b_horizontal_subpanel_src_rank = vertical_neighbors[b_horizontal_subpanel_src_rank_index];

    RANK_PRINTF(xx, "a_vertical_subpanel_src_rank_index = %i\n",    a_vertical_subpanel_src_rank_index);
    RANK_PRINTF(xx, "b_horizontal_subpanel_src_rank_index = %i\n",  b_horizontal_subpanel_src_rank_index);
    RANK_PRINTF(xx, "a_vertical_subpanel_src_rank = %i\n",          a_vertical_subpanel_src_rank);
    RANK_PRINTF(xx, "b_horizontal_subpanel_src_rank = %i\n",        b_horizontal_subpanel_src_rank);
    RANK_PRINTF(xx, "pb_k=%i, k=%i, m=%i, procGridX=%i, procGridY=%i, pb=%i\n", pb_k, k, m,
        procGridX, procGridY, pb);

#ifdef PDEBUG
    fflush(stderr);
    MPI_Barrier(MPI_COMM_WORLD);
    fflush(stderr);
#endif

    assert(a_vertical_subpanel_src_rank < procGridY);
    assert(b_horizontal_subpanel_src_rank < procGridX);

    a_subpanel_transmit_ready = (0 == ((pb_k+1) % a_block_num_columns));
    b_subpanel_transmit_ready = (0 == ((pb_k+1) % b_block_num_rows));

    RANK_PRINTF(xx, "a_subpanel_transmit_ready=%i\n", a_subpanel_transmit_ready);
    RANK_PRINTF(xx, "b_subpanel_transmit_ready=%i\n", b_subpanel_transmit_ready);

    if (rank == a_vertical_subpanel_src_rank)
    {
      /* I'm the horizontal neighbor source */
      RANK_PRINTF(xx, "I am the source of the Asubpanel\n");
      RANK_PRINTF(xx, "Sourcing starting from column %i of my Asubpanel\n", subpanel_start_column);
      RANK_PRINTF(xx, "Ablock (source)\n");
      RANK_FN(xx, print_matrix, a_block_num_rows, a_block_num_columns, Ablock);

      /* Each loop of pb_k, copy a single column. */

      /* Populate Apanel; a_block_num_rows by pb matrix */
      for (c = 0; c < 1; c++)
      {
        for (r = 0; r < a_block_num_rows; r++)
        {
          int ablock_index = r + subpanel_start_column*a_block_num_rows + c*a_block_num_rows;
          int asubpanel_index = r + c*a_block_num_rows + (pb_k%a_block_num_columns)*a_block_num_rows;

          Asubpanel[asubpanel_index] = Ablock[ablock_index];

          assert(asubpanel_index < a_block_num_columns * a_block_num_rows);
          assert(ablock_index < a_block_num_columns * a_block_num_rows);
        }
      }
    }

    if (rank == b_horizontal_subpanel_src_rank)
    {
      int subpanel_start_row = pb_k % b_block_num_rows;

      /* I'm the vertical neighbor source */
      RANK_PRINTF(xx, "I am the source of the Bsubpanel\n");

      RANK_PRINTF(xx, "Sourcing starting from row %i of my Bpanel\n", subpanel_start_row);
      RANK_PRINTF(xx, "Bblock (source)\n");
      RANK_FN(xx, print_matrix, b_block_num_rows, b_block_num_columns, Bblock);

      /* Populate Bpanel; pb by b_block_num_columns matrix */
      for (c = 0; c < b_block_num_columns; c++)
      {
        for (r = 0; r < 1; r++)
        {
          int bblock_index = subpanel_start_row + r + c*b_block_num_rows;
          int bsubpanel_index = r + c*b_block_num_rows + (pb_k%b_block_num_rows);

          Bsubpanel[bsubpanel_index] = Bblock[bblock_index];

          assert(bsubpanel_index < b_block_num_columns * b_block_num_rows);
          assert(bblock_index < b_block_num_columns * b_block_num_rows);
        }
      }

      RANK_PRINTF(xx, "Bsubpanel (before potential transmit; transmit ready=%i)\n",
          b_subpanel_transmit_ready);
      RANK_FN(xx, print_matrix, b_block_num_rows, b_block_num_columns, Bsubpanel);
    }

    if (a_subpanel_transmit_ready)
    {
      MPI_Bcast(Asubpanel, a_block_num_columns * a_block_num_rows, MPI_DOUBLE,
          a_vertical_subpanel_src_rank_index, horizontal_comm);
    }

    if (b_subpanel_transmit_ready)
    {
      MPI_Bcast(Bsubpanel, b_block_num_rows * b_block_num_columns, MPI_DOUBLE,
          b_horizontal_subpanel_src_rank_index, vertical_comm);
    }

    if (a_subpanel_transmit_ready)
    {
      if (rank != a_vertical_subpanel_src_rank)
      {
        RANK_PRINTF(xx, "Asubpanel **received** (%i <-- %i [index %i])\n", xx,
            a_vertical_subpanel_src_rank, a_vertical_subpanel_src_rank_index);
        RANK_FN(xx, print_matrix, a_block_num_rows, a_block_num_columns, Asubpanel);
      }

      /* Accumulate into Apanel */
      for (c = 0; c < a_block_num_columns; c++)
      {
        for (r = 0; r < a_block_num_rows; r++)
        {
          int asubpanel_index = r + c*a_block_num_rows;
          int apanel_index = r + c*a_block_num_rows + (pb_k-a_block_num_columns+1)%pb * a_block_num_rows;

          Apanel[apanel_index] = Asubpanel[asubpanel_index];

          assert(asubpanel_index < a_block_num_columns * a_block_num_rows);
          assert(ablock_index < a_block_num_columns * a_block_num_rows);
        }
      }

      RANK_PRINTF(xx, "Accumulated Apanel\n");
      RANK_FN(xx, print_matrix, a_block_num_rows, pb, Apanel);
    }

    if (b_subpanel_transmit_ready)
    {
      if (rank != b_horizontal_subpanel_src_rank)
      {
        RANK_PRINTF(xx, "Bsubpanel **received** (%i <-- %i [index %i])\n", xx,
            b_horizontal_subpanel_src_rank, b_horizontal_subpanel_src_rank_index);
        RANK_FN(xx, print_matrix, b_block_num_rows, b_block_num_columns, Bsubpanel);
      }

      /* Accumulate into Bpanel */
      for (c = 0; c < b_block_num_columns; c++)
      {
        int row_addition_factor = (pb_k-b_block_num_rows+1)%pb;
        int column_addition_factor = c*pb;
        RANK_PRINTF(xx, "row_addition_factor = %i\n", row_addition_factor);
        RANK_PRINTF(xx, "column_addition_factor = %i\n", column_addition_factor);

        for (r = 0; r < b_block_num_rows; r++)
        {
          int bsubpanel_index = r + c*b_block_num_rows;
          int bpanel_index = r%b_block_num_rows + row_addition_factor + column_addition_factor;

          Bpanel[bpanel_index] = Bsubpanel[bsubpanel_index];

          assert(bsubpanel_index < b_block_num_columns * b_block_num_rows);
          assert(bblock_index < b_block_num_columns * b_block_num_rows);
        }
      }

      RANK_PRINTF(xx, "Accumulated Bpanel\n");
      RANK_FN(xx, print_matrix, pb, b_block_num_columns, Bpanel);
    }

#ifdef PDEBUG
    fflush(stderr);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (0 == ((pb_k+1) % pb))
    {
      RANK_PRINTF(xx, "Performing local_mm()\n");

      local_mm(a_block_num_rows, b_block_num_columns, pb, 1.0,
          Apanel, a_block_num_rows, Bpanel, pb,
          1.0, Cblock, a_block_num_rows);
    }
  }

  /* Free up allocated resources */
  MPI_Comm_free(&vertical_comm);
  MPI_Group_free(&vertical_group);

  MPI_Comm_free(&horizontal_comm);
  MPI_Group_free(&horizontal_group);

  free(horizontal_neighbors);
  free(vertical_neighbors);
  free(Bpanel);
  free(Apanel);

	MPI_Barrier(MPI_COMM_WORLD);
}

void summa(int m, int n, int k, double *Ablock, double *Bblock, double *Cblock,
		int procGridX, int procGridY, int pb)
{
  int rank;

  int a_block_num_columns;
  int a_block_num_rows;
  int b_block_num_columns;
  int b_block_num_rows;

  a_block_num_columns = k/procGridY;
  a_block_num_rows = m/procGridX;
  b_block_num_columns = n/procGridY;
  b_block_num_rows = k/procGridX;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef NONO
  if (a_block_num_columns <= pb)
  {
    RANK_PRINTF(0, "Using SUMMA for large block size\n");
    summa_large(m, n, k, Ablock, Bblock, Cblock, procGridX, procGridY, pb);
  }
  else
  {
#endif
    RANK_PRINTF(0, "Using SUMMA for small block size\n");
    summa_small(m, n, k, Ablock, Bblock, Cblock, procGridX, procGridY, pb);
#ifdef NONO
  }
#endif
}
