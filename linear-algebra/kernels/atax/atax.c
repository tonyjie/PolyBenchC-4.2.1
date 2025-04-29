/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* atax.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "atax.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;
  DATA_TYPE fn;
  fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
      x[i] = 1 + (i / fn);
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % n) / (5*m);
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
void kernel_atax(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(tmp,M,m))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    y[i] = 0;
  for (i = 0; i < _PB_M; i++)
    {
      tmp[i] = SCALAR_VAL(0.0);
      for (j = 0; j < _PB_N; j++)
	tmp[i] = tmp[i] + A[i][j] * x[j];
      for (j = 0; j < _PB_N; j++)
	y[j] = y[j] + A[i][j] * tmp[i];
    }
#pragma endscop

}

/* Intel MKL optimized ATAX kernel using GEMV routines */
static
void kernel_atax_mkl(int m, int n,
                   DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
                   DATA_TYPE POLYBENCH_1D(x,N,n),
                   DATA_TYPE POLYBENCH_1D(y,N,n),
                   DATA_TYPE POLYBENCH_1D(tmp,M,m))
{
#ifdef DATA_TYPE_IS_DOUBLE
  const double alpha = 1.0;
  const double beta = 0.0;
  
  /* Initialize y to zero */
  for (int i = 0; i < n; i++)
    y[i] = 0.0;
    
  /* tmp = A * x (matrix-vector multiply) */
  cblas_dgemv(CblasRowMajor,    /* Matrix layout: Row-major */
              CblasNoTrans,      /* No transpose of matrix A */
              m, n,              /* Matrix dimensions */
              alpha,             /* Alpha scalar */
              &A[0][0], n,       /* Matrix A and leading dimension */
              x, 1,              /* Vector x and stride */
              beta,              /* Beta scalar */
              tmp, 1);           /* Result vector tmp and stride */
              
  /* For each row i, compute y += A[i,*]' * tmp[i] */
  for (int i = 0; i < m; i++) {
    /* y += A[i,:] * tmp[i] (scale row i of A by tmp[i] and add to y) */
    cblas_daxpy(n,               /* Vector length */
                tmp[i],          /* Scalar alpha (tmp[i]) */
                &A[i][0], 1,     /* Row i of A and stride */
                y, 1);           /* Vector y and stride */
  }
  
#elif defined(DATA_TYPE_IS_FLOAT)
  const float alpha = 1.0f;
  const float beta = 0.0f;
  
  /* Initialize y to zero */
  for (int i = 0; i < n; i++)
    y[i] = 0.0f;
    
  /* tmp = A * x (matrix-vector multiply) */
  cblas_sgemv(CblasRowMajor,    /* Matrix layout: Row-major */
              CblasNoTrans,      /* No transpose of matrix A */
              m, n,              /* Matrix dimensions */
              alpha,             /* Alpha scalar */
              &A[0][0], n,       /* Matrix A and leading dimension */
              x, 1,              /* Vector x and stride */
              beta,              /* Beta scalar */
              tmp, 1);           /* Result vector tmp and stride */
              
  /* For each row i, compute y += A[i,*]' * tmp[i] */
  for (int i = 0; i < m; i++) {
    /* y += A[i,:] * tmp[i] (scale row i of A by tmp[i] and add to y) */
    cblas_saxpy(n,               /* Vector length */
                tmp[i],          /* Scalar alpha (tmp[i]) */
                &A[i][0], 1,     /* Row i of A and stride */
                y, 1);           /* Vector y and stride */
  }
#else
  /* For integer types, we fall back to the naive implementation */
  kernel_atax(m, n, A, x, y, tmp);
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
double time_kernel(void (*kernel)(int, int, 
                                 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
                                 DATA_TYPE POLYBENCH_1D(x,N,n),
                                 DATA_TYPE POLYBENCH_1D(y,N,n),
                                 DATA_TYPE POLYBENCH_1D(tmp,M,m)),
                  int m, int n,
                  DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
                  DATA_TYPE POLYBENCH_1D(x,N,n),
                  DATA_TYPE POLYBENCH_1D(y,N,n),
                  DATA_TYPE POLYBENCH_1D(tmp,M,m))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(m, n, A, x, y, tmp);
    
    /* End timer */
    end_time = get_time();
    
    return end_time - start_time;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, N, m, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, M, m);
  
  /* For MKL implementation */
  POLYBENCH_1D_ARRAY_DECL(y_mkl, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(tmp_mkl, DATA_TYPE, M, m);

  /* Initialize array(s). */
  init_array (m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  /* Time naive implementation */
  double naive_time = time_kernel(kernel_atax, m, n,
                                POLYBENCH_ARRAY(A),
                                POLYBENCH_ARRAY(x),
                                POLYBENCH_ARRAY(y),
                                POLYBENCH_ARRAY(tmp));
  
  printf("Naive ATAX Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_atax_mkl, m, n,
                              POLYBENCH_ARRAY(A),
                              POLYBENCH_ARRAY(x),
                              POLYBENCH_ARRAY(y_mkl),
                              POLYBENCH_ARRAY(tmp_mkl));
  
  printf("MKL ATAX Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(n, POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(y_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(y_mkl);
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(tmp_mkl);

  return 0;
}
