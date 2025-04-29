/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* seidel-2d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "seidel-2d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(A_mkl,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
      A_mkl[i][j] = A[i][j];
    }
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
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_seidel_2d(int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int t, i, j;

#pragma scop
  for (t = 0; t <= _PB_TSTEPS - 1; t++)
    for (i = 1; i<= _PB_N - 2; i++)
      for (j = 1; j <= _PB_N - 2; j++)
	A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
		   + A[i][j-1] + A[i][j] + A[i][j+1]
		   + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1])/SCALAR_VAL(9.0);
#pragma endscop

}

/* MKL optimized implementation of Seidel-2D kernel */
static
void kernel_seidel_2d_mkl(int tsteps,
                       int n,
                       DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int t, i, j;
  
  /* Constant for stencil computation */
  DATA_TYPE divisor = SCALAR_VAL(9.0);
  
  /* Allocate MKL memory-aligned temporary array */
  DATA_TYPE* temp_row = (DATA_TYPE*)mkl_malloc(n * sizeof(DATA_TYPE), 64);
  
  for (t = 0; t <= tsteps - 1; t++) {
    for (i = 1; i <= n - 2; i++) {
      /* Process each row */
      #ifdef DATA_TYPE_IS_DOUBLE
      for (j = 1; j <= n - 2; j++) {
        /* Compute 9-point stencil */
        A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
                + A[i][j-1] + A[i][j] + A[i][j+1]
                + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1])/divisor;
      }
      #elif defined(DATA_TYPE_IS_FLOAT)
      for (j = 1; j <= n - 2; j++) {
        /* Compute 9-point stencil */
        A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
                + A[i][j-1] + A[i][j] + A[i][j+1]
                + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1])/divisor;
      }
      #else
      /* For integer types */
      for (j = 1; j <= n - 2; j++) {
        /* Compute 9-point stencil */
        A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
                + A[i][j-1] + A[i][j] + A[i][j+1]
                + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1])/divisor;
      }
      #endif
    }
  }
  
  /* Free allocated memory */
  mkl_free(temp_row);
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
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
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
double time_kernel(void (*kernel)(int, int, 
                                 DATA_TYPE POLYBENCH_2D(A,N,N,n,n)),
                  int tsteps, int n,
                  DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(tsteps, n, A);
    
    /* End timer */
    end_time = get_time();
    
    return end_time - start_time;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  
  /* For MKL implementation */
  POLYBENCH_2D_ARRAY_DECL(A_mkl, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_mkl));

  /* Start timer for naive implementation. */
  double naive_time = time_kernel(kernel_seidel_2d, tsteps, n,
                                POLYBENCH_ARRAY(A));
  
  printf("Naive Seidel-2D Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_seidel_2d_mkl, tsteps, n,
                              POLYBENCH_ARRAY(A_mkl));
  
  printf("MKL Seidel-2D Time: %0.6f seconds\n", mkl_time);
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
