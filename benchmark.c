#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <math.h>

#include "matrix_utils.h"
#include "matmult.h"
#include "stopwatch.h"
#include "square_dgemm.h"



/*
 Matrix sizes used for benchmarking
*/

#ifdef FAST
const int test_sizes[] = {
  2040, 2048
};
#else
const int test_sizes[] = {
 3968, 4096, 4104, 4608, 5120 
};
#endif



/*
 Matrix sizes used for validating the implementation
*/

#ifdef FAST
const int validate_sizes[] = {
  256, 262, 512, 1024, 2040, 2040 
};
#else
const int validate_sizes[] = {
  256, 262, 512, 1024, 3960, 3968, 4080, 4088, 4096, 4104, 
  4160, 4176, 4608, 5120, 5128
};
#endif


#define GIGA 1.0e9
#define N_RUNS 2 //10
#define N_PASSES 2
#define MIN_SIZE 256
#define MAX_SIZE 8192 
#define N_TESTS (sizeof (test_sizes) / sizeof (int))
#define N_VALIDATES (sizeof (validate_sizes) / sizeof (int))


/*
 We create two of each so that we can alternate matrices
 as a way of flushing the cache
*/

static struct stopwatch_t *timer;
static double *A1, *A2, *B1, *B2, *C1, *C2;


/* Forward declarations */
void benchmark_init();
void benchmark_cleanup();
double time_matmult(const int n);
bool validate_matmult(const int n);


int
main() {
	int test;
	int pass;
	double perf_total = 0.0;

	benchmark_init();

	/* Validate the implementation */
	printf("\n\n**Validating matmult**\n");

	for (test=0; test < (int) N_VALIDATES; test++) {
		int n = validate_sizes[test];
		//if (validate_matmult(n) == false) {
		if (validate_matmult(n) == false) {
			printf("\tValidating n=%d.....fail\n",n);
		} else {
			printf("\tValidating n=%d.....pass\n",n);
		}
	}


	/* Time the implementation */
	printf("\n\n**Timing matmult**\n");

	for (pass = 0; pass < N_PASSES; pass++) {
	 for (test=0; test < (int) N_TESTS; test++) {
		int n = test_sizes[test];
		double gflops = time_matmult(n) / GIGA;
		
		printf("\tn=%d: %f gflops/s\n",n,gflops);
		perf_total += gflops;
	 }
	 printf("\n");
	}

	printf("\n\nAverage Performance: %f gflops/s\n",perf_total / N_TESTS / N_PASSES);



	benchmark_cleanup();

	return 0;	
}

void 
benchmark_init() {

	A1 = (double *) malloc(sizeof(double) * MAX_SIZE * MAX_SIZE);
	A2 = (double *) malloc(sizeof(double) * MAX_SIZE * MAX_SIZE);
	B1 = (double *) malloc(sizeof(double) * MAX_SIZE * MAX_SIZE);
	B2 = (double *) malloc(sizeof(double) * MAX_SIZE * MAX_SIZE);
	C1 = (double *) malloc(sizeof(double) * MAX_SIZE * MAX_SIZE);
	C2 = (double *) malloc(sizeof(double) * MAX_SIZE * MAX_SIZE);


	if (A1 == NULL || A2 == NULL || B1 == NULL || B2 == NULL || C1 == NULL || C2 == NULL ) {
		printf("Error: Could not allocate matrices\n");
		exit(-1);
	}

	stopwatch_init();
	timer = stopwatch_create();
}

void 
benchmark_cleanup() {

	free(A1);
	free(A2);
	free(B1);
	free(B2);
	free(C1);
	free(C2);
	
	stopwatch_destroy(timer);
}



bool
validate_matmult(const int n) {
	
  random_matrix(n,n,A1);
  random_matrix(n,n,B1);
  zero_matrix(n,n,C1);

  copy_matrix(n,n,A1,A2);
  copy_matrix(n,n,B1,B2);
  copy_matrix(n,n,C1,C2);

  square_dgemm(n, A1, B1, C1);
  matmult(n, A2, B2, C2);

	
	return verify_matrix(n,n,C1,C2);
}


double 
time_matmult(const int n) {
	
	int i;
	long double flops;
	long double t;
	long double ops = (long double) 2.0 * (long double) n * (long double) n * (long double) n;

	random_matrix(n,n,A1);
 	random_matrix(n,n,B1);
 	zero_matrix(n,n,C1);

	random_matrix(n,n,A2);
 	random_matrix(n,n,B2);
 	zero_matrix(n,n,C2);


	//two warmup runs
	for (i = 0; i < 2; i++) {
		matmult(n, A1, B1, C1);
		matmult(n, A2, B2, C2);
	}

	/* Start the timer */
	stopwatch_start(timer);

	for (i = 0; i < N_RUNS/2; i++) {
		/* Alternate matrices to clear the cache */
		matmult(n, A1, B1, C1);
		matmult(n, A2, B2, C2);
	}

	/* Stop the timer */
	stopwatch_stop(timer);


	t = stopwatch_elapsed(timer) / (long double) N_RUNS;
	//printf("\nOperation took: %Lf sec\n",t);
	flops = (long double) ops / t;

	return (double) flops;
}
