/**
 *  \file matrix_utils.c
 *  \brief matrix utilities
 *  \author Kent Czechowski <kentcz@gatech...>
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>

#include "matrix_utils.h"

#define true 1
#define false 0
#define bool char

#define EPS 0.0001

/** 
 * Similar to verify_matrix(),
 *  this function verifies that each element of A
 *  matches the corresponding element of B
 *
 *  returns true if A and B are equal
 */
bool verify_matrix(int m, int n, double *A, double *B) {

  /* Loop over every element of A and B */
  int row, col;
  for (col = 0; col < n; col++) {
    for (row = 0; row < m; row++) {
      int index = (col * m) + row;
      double a = A[index];
      double b = B[index];

      if (a < b - EPS) {
        return false;
      }
      if (a > b + EPS) {
        return false;
      }
    } /* row */
  }/* col */

  return true;
}


/**
 * Print the elements of the matrix
 **/
void print_matrix(int rows, int cols, double *mat) {

  int r, c;

  /* Iterate over the rows of the matrix */
  for (r = 0; r < rows; r++) {
    /* Iterate over the columns of the matrix */
    for (c = 0; c < cols; c++) {
      int index = (c * rows) + r;
      printf("%.1lf ", mat[index]);
    } /* c */
    printf("\n");
  } /* r */
}

/**
 * Set the elements of the matrix to random values
 **/
void random_matrix(int rows, int cols, double *mat) {

  int r, c;

  /* Iterate over the columns of the matrix */
  for (c = 0; c < cols; c++) {
    /* Iterate over the rows of the matrix */
    for (r = 0; r < rows; r++) {
      int index = (c * rows) + r;
      mat[index] = round(10.0 * rand() / (RAND_MAX + 1.0));
    } /* r */
  } /* c */
}

/**
 * Set the elements of the matrix to zero
 **/
void zero_matrix(int rows, int cols, double *mat) {

  int r, c;

  /* Iterate over the columns of the matrix */
  for (c = 0; c < cols; c++) {
    /* Iterate over the rows of the matrix */
    for (r = 0; r < rows; r++) {
      int index = (c * rows) + r;
      mat[index] = 0.0;
    } /* r */
  } /* c */
}

/**
 * Copy elements of src to dest
 **/
void copy_matrix(int rows, int cols, double *src, double *dest) {

  int r, c;

  /* Iterate over the columns of the matrix */
  for (c = 0; c < cols; c++) {
    /* Iterate over the rows of the matrix */
    for (r = 0; r < rows; r++) {
      int index = (c * rows) + r;
      dest[index] = src[index];
    } /* r */
  } /* c */
}


