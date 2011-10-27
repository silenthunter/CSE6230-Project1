/**
 *  \file local_mm_wrapper.c
 *  \brief C interface for when local_mm is implemented in Fortran
 *  \author Kent Czechowski <kentcz@gatech...>
 */

#include <stdlib.h>
#include <stdio.h>

extern void matmult_(const int n, const double *A, const double *B, double *C);


void matmult(const int n, const double *A, const double *B, double *C) {
  matmult_(&n,A,B,C);
  return;

}
