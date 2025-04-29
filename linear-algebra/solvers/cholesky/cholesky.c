/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* cholesky.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "cholesky.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(A_mkl,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
	A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++) {
	A[i][j] = 0;
      }
      A[i][i] = 1;
    }

  /* Make the matrix positive semi-definite. */
  int r,s,t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      (POLYBENCH_ARRAY(B))[r][s] = 0;
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	(POLYBENCH_ARRAY(B))[r][s] += A[r][t] * A[s][t];
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	A[r][s] = (POLYBENCH_ARRAY(B))[r][s];
  POLYBENCH_FREE_ARRAY(B);
  
  /* Copy A to A_mkl */
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A_mkl[i][j] = A[i][j];
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j <= i; j++) {
    if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
  }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_cholesky(int n,
		     DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j, k;


#pragma scop
  for (i = 0; i < _PB_N; i++) {
     //j<i
     for (j = 0; j < i; j++) {
        for (k = 0; k < j; k++) {
           A[i][j] -= A[i][k] * A[j][k];
        }
        A[i][j] /= A[j][j];
     }
     // i==j case
     for (k = 0; k < i; k++) {
        A[i][i] -= A[i][k] * A[i][k];
     }
     A[i][i] = SQRT_FUN(A[i][i]);
  }
#pragma endscop

}

/* Intel MKL optimized Cholesky kernel */
static
void kernel_cholesky_mkl(int n,
                       DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
#ifdef DATA_TYPE_IS_DOUBLE
  /* Use MKL's LAPACK dpotrf function for Cholesky decomposition */
  int info;
  char uplo = 'L'; /* Compute the lower triangular part */
  
  /* Note: LAPACK functions work with column-major matrices,
     but we have row-major matrices, so we ask for the lower
     triangular part but get the upper triangular part transposed */
  dpotrf(&uplo, &n, &A[0][0], &n, &info);
  
  /* Handle non-positive definite matrices (this shouldn't happen with our initialization) */
  if (info != 0) {
    printf("Warning: dpotrf returned info = %d (matrix may not be positive definite)\n", info);
    /* Fall back to naive implementation */
    kernel_cholesky(n, A);
  }
  
#elif defined(DATA_TYPE_IS_FLOAT)
  /* Use MKL's LAPACK spotrf function for Cholesky decomposition with single precision */
  int info;
  char uplo = 'L'; /* Compute the lower triangular part */
  
  spotrf(&uplo, &n, &A[0][0], &n, &info);
  
  if (info != 0) {
    printf("Warning: spotrf returned info = %d (matrix may not be positive definite)\n", info);
    /* Fall back to naive implementation */
    kernel_cholesky(n, A);
  }
#else
  /* For integer types, we fall back to the naive implementation */
  kernel_cholesky(n, A);
#endif
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int n,
                 DATA_TYPE POLYBENCH_2D(A_naive,N,N,n,n),
                 DATA_TYPE POLYBENCH_2D(A_mkl,N,N,n,n))
{
    int i, j;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    /* Only check the lower triangular part, which is what Cholesky computes */
    for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++) {
            diff = fabs(A_naive[i][j] - A_mkl[i][j]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    
    printf("Maximum difference between naive and MKL implementation: %e\n", max_diff);
    
    if (max_diff < threshold) {
        printf("Verification PASSED: Results match within threshold\n");
        return 1; /* Success */
    } else {
        printf("Verification FAILED: Results differ beyond threshold\n");
        return 0; /* Failure */
    }
}

/* Timer function - uses high resolution timer if available */
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Function to time the execution of a kernel */
static
double time_kernel(void (*kernel)(int, 
                                 DATA_TYPE POLYBENCH_2D(A,N,N,n,n)),
                  int n,
                  DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(n, A);
    
    /* End timer */
    end_time = get_time();
    
    return end_time - start_time;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  
  /* For MKL implementation */
  POLYBENCH_2D_ARRAY_DECL(A_mkl, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_mkl));

  /* Time naive implementation */
  double naive_time = time_kernel(kernel_cholesky, n, POLYBENCH_ARRAY(A));
  
  printf("Naive Cholesky Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_cholesky_mkl, n, POLYBENCH_ARRAY(A_mkl));
  
  printf("MKL Cholesky Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(A_mkl);

  return 0;
}
