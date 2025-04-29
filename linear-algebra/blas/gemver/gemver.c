/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemver.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemver.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE *alpha,
		 DATA_TYPE *beta,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(u1,N,n),
		 DATA_TYPE POLYBENCH_1D(v1,N,n),
		 DATA_TYPE POLYBENCH_1D(u2,N,n),
		 DATA_TYPE POLYBENCH_1D(v2,N,n),
		 DATA_TYPE POLYBENCH_1D(w,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(z,N,n),
        DATA_TYPE POLYBENCH_2D(A_mkl,N,N,n,n),
        DATA_TYPE POLYBENCH_1D(w_mkl,N,n),
        DATA_TYPE POLYBENCH_1D(x_mkl,N,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;

  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      u1[i] = i;
      u2[i] = ((i+1)/fn)/2.0;
      v1[i] = ((i+1)/fn)/4.0;
      v2[i] = ((i+1)/fn)/6.0;
      y[i] = ((i+1)/fn)/8.0;
      z[i] = ((i+1)/fn)/9.0;
      x[i] = 0.0;
      w[i] = 0.0;
      x_mkl[i] = 0.0;
      w_mkl[i] = 0.0;
      for (j = 0; j < n; j++)
      {
        A[i][j] = (DATA_TYPE) (i*j % n) / n;
        A_mkl[i][j] = A[i][j];
      }
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(w,N,n))
{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("w");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, w[i]);
  }
  POLYBENCH_DUMP_END("w");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemver(int n,
		   DATA_TYPE alpha,
		   DATA_TYPE beta,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(u1,N,n),
		   DATA_TYPE POLYBENCH_1D(v1,N,n),
		   DATA_TYPE POLYBENCH_1D(u2,N,n),
		   DATA_TYPE POLYBENCH_1D(v2,N,n),
		   DATA_TYPE POLYBENCH_1D(w,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n),
		   DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

#pragma scop

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < _PB_N; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      w[i] = w[i] +  alpha * A[i][j] * x[j];

#pragma endscop
}

/* Intel MKL optimized GEMVER kernel */
static
void kernel_gemver_mkl(int n,
                  DATA_TYPE alpha,
                  DATA_TYPE beta,
                  DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
                  DATA_TYPE POLYBENCH_1D(u1,N,n),
                  DATA_TYPE POLYBENCH_1D(v1,N,n),
                  DATA_TYPE POLYBENCH_1D(u2,N,n),
                  DATA_TYPE POLYBENCH_1D(v2,N,n),
                  DATA_TYPE POLYBENCH_1D(w,N,n),
                  DATA_TYPE POLYBENCH_1D(x,N,n),
                  DATA_TYPE POLYBENCH_1D(y,N,n),
                  DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i;
  
  /* Step 1: A = A + u1 * v1' + u2 * v2' */
  /* First, compute A = A + u1 * v1' using MKL's ?ger (rank-1 update) */
#ifdef DATA_TYPE_IS_DOUBLE
  cblas_dger(CblasRowMajor, n, n, 1.0, u1, 1, v1, 1, &A[0][0], n);
  
  /* Then, compute A = A + u2 * v2' */
  cblas_dger(CblasRowMajor, n, n, 1.0, u2, 1, v2, 1, &A[0][0], n);
  
  /* Step 2: x = beta * A' * y + x */
  /* Using MKL's ?gemv (matrix-vector multiplication) */
  cblas_dgemv(CblasRowMajor, CblasTrans, n, n, beta, &A[0][0], n, y, 1, 1.0, x, 1);
  
  /* Step 3: x = x + z */
  cblas_daxpy(n, 1.0, z, 1, x, 1);
  
  /* Step 4: w = alpha * A * x + w */
  cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, alpha, &A[0][0], n, x, 1, 1.0, w, 1);
#elif defined(DATA_TYPE_IS_FLOAT)
  cblas_sger(CblasRowMajor, n, n, 1.0f, u1, 1, v1, 1, &A[0][0], n);
  cblas_sger(CblasRowMajor, n, n, 1.0f, u2, 1, v2, 1, &A[0][0], n);
  cblas_sgemv(CblasRowMajor, CblasTrans, n, n, beta, &A[0][0], n, y, 1, 1.0f, x, 1);
  cblas_saxpy(n, 1.0f, z, 1, x, 1);
  cblas_sgemv(CblasRowMajor, CblasNoTrans, n, n, alpha, &A[0][0], n, x, 1, 1.0f, w, 1);
#else
  /* For integer types, fall back to the naive implementation */
  kernel_gemver(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
#endif
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int n,
                 DATA_TYPE POLYBENCH_1D(w_naive,N,n),
                 DATA_TYPE POLYBENCH_1D(w_mkl,N,n))
{
    int i;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < n; i++) {
        diff = fabs(w_naive[i] - w_mkl[i]);
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
double time_kernel(void (*kernel)(int, DATA_TYPE, DATA_TYPE, 
                                 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
                                 DATA_TYPE POLYBENCH_1D(u1,N,n),
                                 DATA_TYPE POLYBENCH_1D(v1,N,n),
                                 DATA_TYPE POLYBENCH_1D(u2,N,n),
                                 DATA_TYPE POLYBENCH_1D(v2,N,n),
                                 DATA_TYPE POLYBENCH_1D(w,N,n),
                                 DATA_TYPE POLYBENCH_1D(x,N,n),
                                 DATA_TYPE POLYBENCH_1D(y,N,n),
                                 DATA_TYPE POLYBENCH_1D(z,N,n)),
                  int n,
                  DATA_TYPE alpha, DATA_TYPE beta,
                  DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
                  DATA_TYPE POLYBENCH_1D(u1,N,n),
                  DATA_TYPE POLYBENCH_1D(v1,N,n),
                  DATA_TYPE POLYBENCH_1D(u2,N,n),
                  DATA_TYPE POLYBENCH_1D(v2,N,n),
                  DATA_TYPE POLYBENCH_1D(w,N,n),
                  DATA_TYPE POLYBENCH_1D(x,N,n),
                  DATA_TYPE POLYBENCH_1D(y,N,n),
                  DATA_TYPE POLYBENCH_1D(z,N,n))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
    
    /* End timer */
    end_time = get_time();
    
    return end_time - start_time;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);
  
  /* Arrays for MKL implementation */
  POLYBENCH_2D_ARRAY_DECL(A_mkl, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(w_mkl, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x_mkl, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(u1),
	      POLYBENCH_ARRAY(v1),
	      POLYBENCH_ARRAY(u2),
	      POLYBENCH_ARRAY(v2),
	      POLYBENCH_ARRAY(w),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y),
	      POLYBENCH_ARRAY(z),
          POLYBENCH_ARRAY(A_mkl),
          POLYBENCH_ARRAY(w_mkl),
          POLYBENCH_ARRAY(x_mkl));

  /* Time naive implementation */
  double naive_time = time_kernel(kernel_gemver, n, alpha, beta,
                                POLYBENCH_ARRAY(A),
                                POLYBENCH_ARRAY(u1),
                                POLYBENCH_ARRAY(v1),
                                POLYBENCH_ARRAY(u2),
                                POLYBENCH_ARRAY(v2),
                                POLYBENCH_ARRAY(w),
                                POLYBENCH_ARRAY(x),
                                POLYBENCH_ARRAY(y),
                                POLYBENCH_ARRAY(z));
  
  printf("Naive GEMVER Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_gemver_mkl, n, alpha, beta,
                              POLYBENCH_ARRAY(A_mkl),
                              POLYBENCH_ARRAY(u1),
                              POLYBENCH_ARRAY(v1),
                              POLYBENCH_ARRAY(u2),
                              POLYBENCH_ARRAY(v2),
                              POLYBENCH_ARRAY(w_mkl),
                              POLYBENCH_ARRAY(x_mkl),
                              POLYBENCH_ARRAY(y),
                              POLYBENCH_ARRAY(z));
  
  printf("MKL GEMVER Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(n, POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(w_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(A_mkl);
  POLYBENCH_FREE_ARRAY(u1);
  POLYBENCH_FREE_ARRAY(v1);
  POLYBENCH_FREE_ARRAY(u2);
  POLYBENCH_FREE_ARRAY(v2);
  POLYBENCH_FREE_ARRAY(w);
  POLYBENCH_FREE_ARRAY(w_mkl);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(x_mkl);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(z);

  return 0;
}
