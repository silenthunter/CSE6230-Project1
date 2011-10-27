#define true 1
#define false 0
#define bool char

/** 
 * Similar to verify_matrix(),
 *  this function verifies that each element of A
 *  matches the corresponding element of B
 *
 *  returns true if A and B are equal
 */
bool verify_matrix(int m, int n, double *A, double *B);

/**
 * Print the elements of the matrix
 **/
void print_matrix(int rows, int cols, double *mat);

/**
 * Set the elements of the matrix to random values
 **/
void random_matrix(int rows, int cols, double *mat);

/**
 * Set the elements of the matrix to zero
 **/
void zero_matrix(int rows, int cols, double *mat);


/**
 * Copy elements of src to dest
 **/
void copy_matrix(int rows, int cols, double *src, double *dest);
