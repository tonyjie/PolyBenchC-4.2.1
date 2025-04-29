/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* mvt.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "mvt.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_1D(x1,N,n),
		DATA_TYPE POLYBENCH_1D(x2,N,n),
		DATA_TYPE POLYBENCH_1D(y_1,N,n),
		DATA_TYPE POLYBENCH_1D(y_2,N,n),
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(x1_mkl,N,n),
		DATA_TYPE POLYBENCH_1D(x2_mkl,N,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      x1[i] = (DATA_TYPE) (i % n) / n;
      x2[i] = (DATA_TYPE) ((i + 1) % n) / n;
      x1_mkl[i] = x1[i];
      x2_mkl[i] = x2[i];
      y_1[i] = (DATA_TYPE) ((i + 3) % n) / n;
      y_2[i] = (DATA_TYPE) ((i + 4) % n) / n;
      for (j = 0; j < n; j++)
	A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x1,N,n),
		 DATA_TYPE POLYBENCH_1D(x2,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x1");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x1[i]);
  }
  POLYBENCH_DUMP_END("x1");

  POLYBENCH_DUMP_BEGIN("x2");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x2[i]);
  }
  POLYBENCH_DUMP_END("x2");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_mvt(int n,
		DATA_TYPE POLYBENCH_1D(x1,N,n),
		DATA_TYPE POLYBENCH_1D(x2,N,n),
		DATA_TYPE POLYBENCH_1D(y_1,N,n),
		DATA_TYPE POLYBENCH_1D(y_2,N,n),
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];
#pragma endscop

}

/* Intel MKL optimized MVT kernel using GEMV routines */
static
void kernel_mvt_mkl(int n,
                  DATA_TYPE POLYBENCH_1D(x1,N,n),
                  DATA_TYPE POLYBENCH_1D(x2,N,n),
                  DATA_TYPE POLYBENCH_1D(y_1,N,n),
                  DATA_TYPE POLYBENCH_1D(y_2,N,n),
                  DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
#ifdef DATA_TYPE_IS_DOUBLE
  const double alpha = 1.0;
  const double beta = 1.0; /* Beta = 1.0 because we're adding to x1/x2 */
  
  /* x1 = x1 + A * y_1 (matrix-vector multiply, no transpose) */
  cblas_dgemv(CblasRowMajor,     /* Matrix storage order: Row-major */
              CblasNoTrans,       /* No transpose of matrix A */
              n, n,               /* Matrix dimensions */
              alpha,              /* Alpha scalar */
              &A[0][0], n,        /* Matrix A and leading dimension */
              y_1, 1,             /* Vector y_1 and stride */
              beta,               /* Beta scalar */
              x1, 1);             /* Result vector x1 and stride */
              
  /* x2 = x2 + A^T * y_2 (matrix-vector multiply with transpose) */
  cblas_dgemv(CblasRowMajor,     /* Matrix storage order: Row-major */
              CblasTrans,         /* Transpose matrix A (use A^T) */
              n, n,               /* Matrix dimensions */
              alpha,              /* Alpha scalar */
              &A[0][0], n,        /* Matrix A and leading dimension */
              y_2, 1,             /* Vector y_2 and stride */
              beta,               /* Beta scalar */
              x2, 1);             /* Result vector x2 and stride */
  
#elif defined(DATA_TYPE_IS_FLOAT)
  const float alpha = 1.0f;
  const float beta = 1.0f; /* Beta = 1.0 because we're adding to x1/x2 */
  
  /* x1 = x1 + A * y_1 (matrix-vector multiply, no transpose) */
  cblas_sgemv(CblasRowMajor,     /* Matrix storage order: Row-major */
              CblasNoTrans,       /* No transpose of matrix A */
              n, n,               /* Matrix dimensions */
              alpha,              /* Alpha scalar */
              &A[0][0], n,        /* Matrix A and leading dimension */
              y_1, 1,             /* Vector y_1 and stride */
              beta,               /* Beta scalar */
              x1, 1);             /* Result vector x1 and stride */
              
  /* x2 = x2 + A^T * y_2 (matrix-vector multiply with transpose) */
  cblas_sgemv(CblasRowMajor,     /* Matrix storage order: Row-major */
              CblasTrans,         /* Transpose matrix A (use A^T) */
              n, n,               /* Matrix dimensions */
              alpha,              /* Alpha scalar */
              &A[0][0], n,        /* Matrix A and leading dimension */
              y_2, 1,             /* Vector y_2 and stride */
              beta,               /* Beta scalar */
              x2, 1);             /* Result vector x2 and stride */
#else
  /* For integer types, we fall back to the naive implementation */
  kernel_mvt(n, x1, x2, y_1, y_2, A);
#endif
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int n,
                 DATA_TYPE POLYBENCH_1D(x1_naive,N,n),
                 DATA_TYPE POLYBENCH_1D(x2_naive,N,n),
                 DATA_TYPE POLYBENCH_1D(x1_mkl,N,n),
                 DATA_TYPE POLYBENCH_1D(x2_mkl,N,n))
{
    int i;
    DATA_TYPE diff;
    DATA_TYPE max_diff_x1 = 0.0;
    DATA_TYPE max_diff_x2 = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < n; i++) {
        diff = fabs(x1_naive[i] - x1_mkl[i]);
        if (diff > max_diff_x1) {
            max_diff_x1 = diff;
        }
        
        diff = fabs(x2_naive[i] - x2_mkl[i]);
        if (diff > max_diff_x2) {
            max_diff_x2 = diff;
        }
    }
    
    printf("Maximum difference in x1 vector: %e\n", max_diff_x1);
    printf("Maximum difference in x2 vector: %e\n", max_diff_x2);
    
    if (max_diff_x1 < threshold && max_diff_x2 < threshold) {
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
                                 DATA_TYPE POLYBENCH_1D(x1,N,n),
                                 DATA_TYPE POLYBENCH_1D(x2,N,n),
                                 DATA_TYPE POLYBENCH_1D(y_1,N,n),
                                 DATA_TYPE POLYBENCH_1D(y_2,N,n),
                                 DATA_TYPE POLYBENCH_2D(A,N,N,n,n)),
                  int n,
                  DATA_TYPE POLYBENCH_1D(x1,N,n),
                  DATA_TYPE POLYBENCH_1D(x2,N,n),
                  DATA_TYPE POLYBENCH_1D(y_1,N,n),
                  DATA_TYPE POLYBENCH_1D(y_2,N,n),
                  DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(n, x1, x2, y_1, y_2, A);
    
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
  POLYBENCH_1D_ARRAY_DECL(x1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_2, DATA_TYPE, N, n);

  /* For MKL implementation */
  POLYBENCH_1D_ARRAY_DECL(x1_mkl, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x2_mkl, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array (n,
	      POLYBENCH_ARRAY(x1),
	      POLYBENCH_ARRAY(x2),
	      POLYBENCH_ARRAY(y_1),
	      POLYBENCH_ARRAY(y_2),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(x1_mkl),
	      POLYBENCH_ARRAY(x2_mkl));

  /* Time naive implementation */
  double naive_time = time_kernel(kernel_mvt, n,
                                POLYBENCH_ARRAY(x1),
                                POLYBENCH_ARRAY(x2),
                                POLYBENCH_ARRAY(y_1),
                                POLYBENCH_ARRAY(y_2),
                                POLYBENCH_ARRAY(A));
  
  printf("Naive MVT Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_mvt_mkl, n,
                              POLYBENCH_ARRAY(x1_mkl),
                              POLYBENCH_ARRAY(x2_mkl),
                              POLYBENCH_ARRAY(y_1),
                              POLYBENCH_ARRAY(y_2),
                              POLYBENCH_ARRAY(A));
  
  printf("MKL MVT Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(n, 
                POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2),
                POLYBENCH_ARRAY(x1_mkl), POLYBENCH_ARRAY(x2_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x1);
  POLYBENCH_FREE_ARRAY(x2);
  POLYBENCH_FREE_ARRAY(x1_mkl);
  POLYBENCH_FREE_ARRAY(x2_mkl);
  POLYBENCH_FREE_ARRAY(y_1);
  POLYBENCH_FREE_ARRAY(y_2);

  return 0;
}
