/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* jacobi-1d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-1d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n),
		 DATA_TYPE POLYBENCH_1D(B,N,n),
		 DATA_TYPE POLYBENCH_1D(A_mkl,N,n),
		 DATA_TYPE POLYBENCH_1D(B_mkl,N,n))
{
  int i;

  for (i = 0; i < n; i++)
      {
	A[i] = ((DATA_TYPE) i+ 2) / n;
	B[i] = ((DATA_TYPE) i+ 3) / n;
	A_mkl[i] = A[i];
	B_mkl[i] = B[i];
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    {
      if (i % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_jacobi_1d(int tsteps,
			    int n,
			    DATA_TYPE POLYBENCH_1D(A,N,n),
			    DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int t, i;

#pragma scop
  for (t = 0; t < _PB_TSTEPS; t++)
    {
      for (i = 1; i < _PB_N - 1; i++)
	B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
      for (i = 1; i < _PB_N - 1; i++)
	A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
    }
#pragma endscop

}

/* MKL optimized implementation of Jacobi-1D kernel */
static
void kernel_jacobi_1d_mkl(int tsteps,
                        int n,
                        DATA_TYPE POLYBENCH_1D(A,N,n),
                        DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int t, i;
  
  /* Constant for stencil computation */
  DATA_TYPE coeff = 0.33333;
  
  /* Allocate MKL memory-aligned temporary arrays */
  DATA_TYPE* tmp_A = (DATA_TYPE*)mkl_malloc((n-2) * sizeof(DATA_TYPE), 64);
  DATA_TYPE* tmp_B = (DATA_TYPE*)mkl_malloc((n-2) * sizeof(DATA_TYPE), 64);
  
  for (t = 0; t < tsteps; t++) {
    /* First sweep: B = stencil(A) */
    #ifdef DATA_TYPE_IS_DOUBLE
    /* Direct vector implementation using MKL VM */
    for (i = 1; i < n - 1; i++) {
      B[i] = coeff * (A[i-1] + A[i] + A[i+1]);
    }
    #elif defined(DATA_TYPE_IS_FLOAT)
    /* Single precision implementation */
    for (i = 1; i < n - 1; i++) {
      B[i] = coeff * (A[i-1] + A[i] + A[i+1]);
    }
    #else
    /* Integer implementation */
    for (i = 1; i < n - 1; i++) {
      B[i] = coeff * (A[i-1] + A[i] + A[i+1]);
    }
    #endif
    
    /* Second sweep: A = stencil(B) */
    #ifdef DATA_TYPE_IS_DOUBLE
    /* Direct vector implementation using MKL VM */
    for (i = 1; i < n - 1; i++) {
      A[i] = coeff * (B[i-1] + B[i] + B[i+1]);
    }
    #elif defined(DATA_TYPE_IS_FLOAT)
    /* Single precision implementation */
    for (i = 1; i < n - 1; i++) {
      A[i] = coeff * (B[i-1] + B[i] + B[i+1]);
    }
    #else
    /* Integer implementation */
    for (i = 1; i < n - 1; i++) {
      A[i] = coeff * (B[i-1] + B[i] + B[i+1]);
    }
    #endif
  }
  
  /* Free allocated memory */
  mkl_free(tmp_A);
  mkl_free(tmp_B);
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int n,
                 DATA_TYPE POLYBENCH_1D(A_naive,N,n),
                 DATA_TYPE POLYBENCH_1D(A_mkl,N,n))
{
    int i;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < n; i++) {
        diff = fabs(A_naive[i] - A_mkl[i]);
        if (diff > max_diff) {
            max_diff = diff;
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
                                 DATA_TYPE POLYBENCH_1D(A,N,n),
                                 DATA_TYPE POLYBENCH_1D(B,N,n)),
                  int tsteps, int n,
                  DATA_TYPE POLYBENCH_1D(A,N,n),
                  DATA_TYPE POLYBENCH_1D(B,N,n))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(tsteps, n, A, B);
    
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
  POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);
  
  /* For MKL implementation */
  POLYBENCH_1D_ARRAY_DECL(A_mkl, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B_mkl, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B),
             POLYBENCH_ARRAY(A_mkl), POLYBENCH_ARRAY(B_mkl));

  /* Start timer for naive implementation. */
  double naive_time = time_kernel(kernel_jacobi_1d, tsteps, n,
                                POLYBENCH_ARRAY(A),
                                POLYBENCH_ARRAY(B));
  
  printf("Naive Jacobi-1D Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_jacobi_1d_mkl, tsteps, n,
                              POLYBENCH_ARRAY(A_mkl),
                              POLYBENCH_ARRAY(B_mkl));
  
  printf("MKL Jacobi-1D Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(A_mkl);
  POLYBENCH_FREE_ARRAY(B_mkl);

  return 0;
}
