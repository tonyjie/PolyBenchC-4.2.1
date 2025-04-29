/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* ludcmp.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "ludcmp.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(b,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_2D(A_mkl,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(b_mkl,N,n),
		 DATA_TYPE POLYBENCH_1D(x_mkl,N,n),
		 DATA_TYPE POLYBENCH_1D(y_mkl,N,n))
{
  int i, j;
  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      x[i] = 0;
      y[i] = 0;
      x_mkl[i] = 0;
      y_mkl[i] = 0;
      b[i] = (i+1)/fn/2.0 + 4;
      b_mkl[i] = b[i];
    }

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
  /* not necessary for LU, but using same code as cholesky */
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
		 DATA_TYPE POLYBENCH_1D(x,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_ludcmp(int n,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(b,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j, k;

  DATA_TYPE w;

#pragma scop
  for (i = 0; i < _PB_N; i++) {
    for (j = 0; j <i; j++) {
       w = A[i][j];
       for (k = 0; k < j; k++) {
          w -= A[i][k] * A[k][j];
       }
        A[i][j] = w / A[j][j];
    }
   for (j = i; j < _PB_N; j++) {
       w = A[i][j];
       for (k = 0; k < i; k++) {
          w -= A[i][k] * A[k][j];
       }
       A[i][j] = w;
    }
  }

  for (i = 0; i < _PB_N; i++) {
     w = b[i];
     for (j = 0; j < i; j++)
        w -= A[i][j] * y[j];
     y[i] = w;
  }

   for (i = _PB_N-1; i >=0; i--) {
     w = y[i];
     for (j = i+1; j < _PB_N; j++)
        w -= A[i][j] * x[j];
     x[i] = w / A[i][i];
  }
#pragma endscop

}

/* Intel MKL optimized implementation */
static
void kernel_ludcmp_mkl(int n,
                      DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
                      DATA_TYPE POLYBENCH_1D(b,N,n),
                      DATA_TYPE POLYBENCH_1D(x,N,n),
                      DATA_TYPE POLYBENCH_1D(y,N,n))
{
    int i, info;
    int* ipiv = (int*)malloc(n * sizeof(int));
    
    /* Copy b to x as we will use x as both input and output vector for MKL */
    for (i = 0; i < n; i++)
        x[i] = b[i];

#ifdef DATA_TYPE_IS_DOUBLE
    /* LU factorization */
    info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, &A[0][0], n, ipiv);
    
    /* Solve the system A*x = b */
    if (info == 0) {
        info = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', n, 1, &A[0][0], n, ipiv, x, 1);
    }
#elif defined(DATA_TYPE_IS_FLOAT)
    /* LU factorization */
    info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, &A[0][0], n, ipiv);
    
    /* Solve the system A*x = b */
    if (info == 0) {
        info = LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', n, 1, &A[0][0], n, ipiv, x, 1);
    }
#else
    /* Fall back to naive implementation for integer types */
    kernel_ludcmp(n, A, b, x, y);
#endif

    free(ipiv);
    
    /* Note: We don't compute y separately in the MKL version as the solve routine directly gives us x */
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
                                 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
                                 DATA_TYPE POLYBENCH_1D(b,N,n),
                                 DATA_TYPE POLYBENCH_1D(x,N,n),
                                 DATA_TYPE POLYBENCH_1D(y,N,n)),
                  int n,
                  DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
                  DATA_TYPE POLYBENCH_1D(b,N,n),
                  DATA_TYPE POLYBENCH_1D(x,N,n),
                  DATA_TYPE POLYBENCH_1D(y,N,n))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(n, A, b, x, y);
    
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
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  
  /* Variables for MKL implementation */
  POLYBENCH_2D_ARRAY_DECL(A_mkl, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(b_mkl, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x_mkl, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_mkl, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array (n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(b),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y),
              POLYBENCH_ARRAY(A_mkl),
              POLYBENCH_ARRAY(b_mkl),
              POLYBENCH_ARRAY(x_mkl),
              POLYBENCH_ARRAY(y_mkl));

  /* Start timer for naive implementation. */
  double naive_time = time_kernel(kernel_ludcmp, n,
                                POLYBENCH_ARRAY(A),
                                POLYBENCH_ARRAY(b),
                                POLYBENCH_ARRAY(x),
                                POLYBENCH_ARRAY(y));
  
  printf("Naive LUDCMP Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_ludcmp_mkl, n,
                              POLYBENCH_ARRAY(A_mkl),
                              POLYBENCH_ARRAY(b_mkl),
                              POLYBENCH_ARRAY(x_mkl),
                              POLYBENCH_ARRAY(y_mkl));
  
  printf("MKL LUDCMP Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(n, POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(x_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(A_mkl);
  POLYBENCH_FREE_ARRAY(b);
  POLYBENCH_FREE_ARRAY(b_mkl);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(x_mkl);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(y_mkl);

  return 0;
}
