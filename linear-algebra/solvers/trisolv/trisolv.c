/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* trisolv.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trisolv.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x,N,n),
		DATA_TYPE POLYBENCH_1D(b,N,n),
		DATA_TYPE POLYBENCH_2D(L_mkl,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x_mkl,N,n),
		DATA_TYPE POLYBENCH_1D(b_mkl,N,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      x[i] = - 999;
      x_mkl[i] = - 999;
      b[i] =  i;
      b_mkl[i] = i;
      for (j = 0; j <= i; j++) {
        L[i][j] = (DATA_TYPE) (i+n-j+1)*2/n;
        L_mkl[i][j] = L[i][j];
      }
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (i = 0; i < n; i++) {
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_trisolv(int n,
		    DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
		    DATA_TYPE POLYBENCH_1D(x,N,n),
		    DATA_TYPE POLYBENCH_1D(b,N,n))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    {
      x[i] = b[i];
      for (j = 0; j <i; j++)
        x[i] -= L[i][j] * x[j];
      x[i] = x[i] / L[i][i];
    }
#pragma endscop

}

/* Intel MKL optimized implementation */
static
void kernel_trisolv_mkl(int n,
                      DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
                      DATA_TYPE POLYBENCH_1D(x,N,n),
                      DATA_TYPE POLYBENCH_1D(b,N,n))
{
    int i;
    char uplo = 'L';  /* Lower triangular matrix */
    char diag = 'N';  /* Non-unit triangular matrix */
    int incx = 1;
    
    /* Copy b to x as LAPACK will overwrite it with the solution */
    for (i = 0; i < n; i++)
        x[i] = b[i];
    
#ifdef DATA_TYPE_IS_DOUBLE
    /* Solve the triangular system using MKL's DTRSV routine */
    cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, 
                n, &L[0][0], n, x, incx);
#elif defined(DATA_TYPE_IS_FLOAT)
    /* Solve the triangular system using MKL's STRSV routine */
    cblas_strsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, 
                n, &L[0][0], n, x, incx);
#else
    /* Fall back to naive implementation for integer types */
    kernel_trisolv(n, L, x, b);
#endif
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int n,
                 DATA_TYPE POLYBENCH_1D(x_naive,N,n),
                 DATA_TYPE POLYBENCH_1D(x_mkl,N,n))
{
    int i;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < n; i++) {
        diff = fabs(x_naive[i] - x_mkl[i]);
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
double time_kernel(void (*kernel)(int, 
                                 DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
                                 DATA_TYPE POLYBENCH_1D(x,N,n),
                                 DATA_TYPE POLYBENCH_1D(b,N,n)),
                  int n,
                  DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
                  DATA_TYPE POLYBENCH_1D(x,N,n),
                  DATA_TYPE POLYBENCH_1D(b,N,n))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(n, L, x, b);
    
    /* End timer */
    end_time = get_time();
    
    return end_time - start_time;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(L, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);
  
  /* Variables for MKL implementation */
  POLYBENCH_2D_ARRAY_DECL(L_mkl, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x_mkl, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(b_mkl, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b),
              POLYBENCH_ARRAY(L_mkl), POLYBENCH_ARRAY(x_mkl), POLYBENCH_ARRAY(b_mkl));

  /* Start timer for naive implementation. */
  double naive_time = time_kernel(kernel_trisolv, n,
                                POLYBENCH_ARRAY(L),
                                POLYBENCH_ARRAY(x),
                                POLYBENCH_ARRAY(b));
  
  printf("Naive TRISOLV Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_trisolv_mkl, n,
                              POLYBENCH_ARRAY(L_mkl),
                              POLYBENCH_ARRAY(x_mkl),
                              POLYBENCH_ARRAY(b_mkl));
  
  printf("MKL TRISOLV Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(n, POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(x_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(L);
  POLYBENCH_FREE_ARRAY(L_mkl);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(x_mkl);
  POLYBENCH_FREE_ARRAY(b);
  POLYBENCH_FREE_ARRAY(b_mkl);

  return 0;
}
