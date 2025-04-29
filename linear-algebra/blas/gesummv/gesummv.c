/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gesummv.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gesummv.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x,N,n),
		DATA_TYPE POLYBENCH_1D(y_mkl,N,n),
		DATA_TYPE POLYBENCH_1D(tmp_mkl,N,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    {
      x[i] = (DATA_TYPE)( i % n) / n;
      for (j = 0; j < n; j++) {
	A[i][j] = (DATA_TYPE) ((i*j+1) % n) / n;
	B[i][j] = (DATA_TYPE) ((i*j+2) % n) / n;
      }
      y_mkl[i] = SCALAR_VAL(0.0);
      tmp_mkl[i] = SCALAR_VAL(0.0);
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(y,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gesummv(int n,
		    DATA_TYPE alpha,
		    DATA_TYPE beta,
		    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		    DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
		    DATA_TYPE POLYBENCH_1D(tmp,N,n),
		    DATA_TYPE POLYBENCH_1D(x,N,n),
		    DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    {
      tmp[i] = SCALAR_VAL(0.0);
      y[i] = SCALAR_VAL(0.0);
      for (j = 0; j < _PB_N; j++)
	{
	  tmp[i] = A[i][j] * x[j] + tmp[i];
	  y[i] = B[i][j] * x[j] + y[i];
	}
      y[i] = alpha * tmp[i] + beta * y[i];
    }
#pragma endscop

}

/* Intel MKL optimized GESUMMV kernel */
static
void kernel_gesummv_mkl(int n,
                    DATA_TYPE alpha,
                    DATA_TYPE beta,
                    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
                    DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
                    DATA_TYPE POLYBENCH_1D(tmp,N,n),
                    DATA_TYPE POLYBENCH_1D(x,N,n),
                    DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i;

#ifdef DATA_TYPE_IS_DOUBLE
  /* tmp = A * x (matrix-vector multiplication) */
  cblas_dgemv(CblasRowMajor, CblasNoTrans, 
              n, n, 1.0, &A[0][0], n, x, 1, 0.0, tmp, 1);
  
  /* y = B * x (matrix-vector multiplication) */
  cblas_dgemv(CblasRowMajor, CblasNoTrans, 
              n, n, 1.0, &B[0][0], n, x, 1, 0.0, y, 1);
  
  /* y = alpha * tmp + beta * y (vector scaling and addition) */
  /* First, scale tmp by alpha and store in a temporary vector */
  for (i = 0; i < n; i++) {
    tmp[i] = alpha * tmp[i];
  }
  
  /* Then, scale y by beta */
  cblas_dscal(n, beta, y, 1);
  
  /* Finally, add tmp to y */
  cblas_daxpy(n, 1.0, tmp, 1, y, 1);
  
#elif defined(DATA_TYPE_IS_FLOAT)
  /* Same operations but with single-precision MKL functions */
  cblas_sgemv(CblasRowMajor, CblasNoTrans, 
              n, n, 1.0f, &A[0][0], n, x, 1, 0.0f, tmp, 1);
  
  cblas_sgemv(CblasRowMajor, CblasNoTrans, 
              n, n, 1.0f, &B[0][0], n, x, 1, 0.0f, y, 1);
  
  for (i = 0; i < n; i++) {
    tmp[i] = alpha * tmp[i];
  }
  
  cblas_sscal(n, beta, y, 1);
  
  cblas_saxpy(n, 1.0f, tmp, 1, y, 1);
#else
  /* Fall back to naive implementation for integer types */
  kernel_gesummv(n, alpha, beta, A, B, tmp, x, y);
#endif
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int n,
                 DATA_TYPE POLYBENCH_1D(y_naive,N,n),
                 DATA_TYPE POLYBENCH_1D(y_mkl,N,n))
{
    int i;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < n; i++) {
        diff = fabs(y_naive[i] - y_mkl[i]);
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
                                 DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
                                 DATA_TYPE POLYBENCH_1D(tmp,N,n),
                                 DATA_TYPE POLYBENCH_1D(x,N,n),
                                 DATA_TYPE POLYBENCH_1D(y,N,n)),
                  int n,
                  DATA_TYPE alpha, DATA_TYPE beta,
                  DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
                  DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
                  DATA_TYPE POLYBENCH_1D(tmp,N,n),
                  DATA_TYPE POLYBENCH_1D(x,N,n),
                  DATA_TYPE POLYBENCH_1D(y,N,n))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(n, alpha, beta, A, B, tmp, x, y);
    
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
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  
  /* Arrays for MKL implementation */
  POLYBENCH_1D_ARRAY_DECL(tmp_mkl, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_mkl, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y_mkl),
	      POLYBENCH_ARRAY(tmp_mkl));

  /* Time naive implementation */
  double naive_time = time_kernel(kernel_gesummv, n, alpha, beta,
                                POLYBENCH_ARRAY(A),
                                POLYBENCH_ARRAY(B),
                                POLYBENCH_ARRAY(tmp),
                                POLYBENCH_ARRAY(x),
                                POLYBENCH_ARRAY(y));
  
  printf("Naive GESUMMV Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_gesummv_mkl, n, alpha, beta,
                              POLYBENCH_ARRAY(A),
                              POLYBENCH_ARRAY(B),
                              POLYBENCH_ARRAY(tmp_mkl),
                              POLYBENCH_ARRAY(x),
                              POLYBENCH_ARRAY(y_mkl));
  
  printf("MKL GESUMMV Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(n, POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(y_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(tmp_mkl);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(y_mkl);

  return 0;
}
