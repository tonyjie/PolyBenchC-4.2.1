/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* bicg.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "bicg.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m))
{
  int i, j;

  for (i = 0; i < m; i++)
    p[i] = (DATA_TYPE)(i % m) / m;
  for (i = 0; i < n; i++) {
    r[i] = (DATA_TYPE)(i % n) / n;
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % n)/n;
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_1D(s,M,m),
		 DATA_TYPE POLYBENCH_1D(q,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("s");
  for (i = 0; i < m; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, s[i]);
  }
  POLYBENCH_DUMP_END("s");
  POLYBENCH_DUMP_BEGIN("q");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, q[i]);
  }
  POLYBENCH_DUMP_END("q");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_bicg(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(s,M,m),
		 DATA_TYPE POLYBENCH_1D(q,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_M; i++)
    s[i] = 0;
  for (i = 0; i < _PB_N; i++)
    {
      q[i] = SCALAR_VAL(0.0);
      for (j = 0; j < _PB_M; j++)
	{
	  s[j] = s[j] + r[i] * A[i][j];
	  q[i] = q[i] + A[i][j] * p[j];
	}
    }
#pragma endscop

}

/* Intel MKL optimized BICG kernel using GEMV routines */
static
void kernel_bicg_mkl(int m, int n,
                   DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
                   DATA_TYPE POLYBENCH_1D(s,M,m),
                   DATA_TYPE POLYBENCH_1D(q,N,n),
                   DATA_TYPE POLYBENCH_1D(p,M,m),
                   DATA_TYPE POLYBENCH_1D(r,N,n))
{
#ifdef DATA_TYPE_IS_DOUBLE
  const double alpha = 1.0;
  const double beta = 0.0;
  
  /* Initialize s to zero */
  for (int i = 0; i < m; i++)
    s[i] = 0.0;
    
  /* s = A^T * r (matrix-vector multiply with transpose) */
  cblas_dgemv(CblasRowMajor,     /* Matrix layout: Row-major */
              CblasTrans,         /* Transpose matrix A (use A^T) */
              n, m,               /* Matrix dimensions (n rows, m cols) */
              alpha,              /* Alpha scalar */
              &A[0][0], m,        /* Matrix A and leading dimension */
              r, 1,               /* Vector r and stride */
              beta,               /* Beta scalar */
              s, 1);              /* Result vector s and stride */
              
  /* q = A * p (matrix-vector multiply without transpose) */
  cblas_dgemv(CblasRowMajor,     /* Matrix layout: Row-major */
              CblasNoTrans,       /* No transpose of matrix A */
              n, m,               /* Matrix dimensions (n rows, m cols) */
              alpha,              /* Alpha scalar */
              &A[0][0], m,        /* Matrix A and leading dimension */
              p, 1,               /* Vector p and stride */
              beta,               /* Beta scalar */
              q, 1);              /* Result vector q and stride */
  
#elif defined(DATA_TYPE_IS_FLOAT)
  const float alpha = 1.0f;
  const float beta = 0.0f;
  
  /* Initialize s to zero */
  for (int i = 0; i < m; i++)
    s[i] = 0.0f;
    
  /* s = A^T * r (matrix-vector multiply with transpose) */
  cblas_sgemv(CblasRowMajor,     /* Matrix layout: Row-major */
              CblasTrans,         /* Transpose matrix A (use A^T) */
              n, m,               /* Matrix dimensions (n rows, m cols) */
              alpha,              /* Alpha scalar */
              &A[0][0], m,        /* Matrix A and leading dimension */
              r, 1,               /* Vector r and stride */
              beta,               /* Beta scalar */
              s, 1);              /* Result vector s and stride */
              
  /* q = A * p (matrix-vector multiply without transpose) */
  cblas_sgemv(CblasRowMajor,     /* Matrix layout: Row-major */
              CblasNoTrans,       /* No transpose of matrix A */
              n, m,               /* Matrix dimensions (n rows, m cols) */
              alpha,              /* Alpha scalar */
              &A[0][0], m,        /* Matrix A and leading dimension */
              p, 1,               /* Vector p and stride */
              beta,               /* Beta scalar */
              q, 1);              /* Result vector q and stride */
#else
  /* For integer types, we fall back to the naive implementation */
  kernel_bicg(m, n, A, s, q, p, r);
#endif
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int m, int n,
                 DATA_TYPE POLYBENCH_1D(s_naive,M,m),
                 DATA_TYPE POLYBENCH_1D(q_naive,N,n),
                 DATA_TYPE POLYBENCH_1D(s_mkl,M,m),
                 DATA_TYPE POLYBENCH_1D(q_mkl,N,n))
{
    int i;
    DATA_TYPE diff;
    DATA_TYPE max_diff_s = 0.0;
    DATA_TYPE max_diff_q = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < m; i++) {
        diff = fabs(s_naive[i] - s_mkl[i]);
        if (diff > max_diff_s) {
            max_diff_s = diff;
        }
    }
    
    for (i = 0; i < n; i++) {
        diff = fabs(q_naive[i] - q_mkl[i]);
        if (diff > max_diff_q) {
            max_diff_q = diff;
        }
    }
    
    printf("Maximum difference in s vector: %e\n", max_diff_s);
    printf("Maximum difference in q vector: %e\n", max_diff_q);
    
    if (max_diff_s < threshold && max_diff_q < threshold) {
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
                                 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
                                 DATA_TYPE POLYBENCH_1D(s,M,m),
                                 DATA_TYPE POLYBENCH_1D(q,N,n),
                                 DATA_TYPE POLYBENCH_1D(p,M,m),
                                 DATA_TYPE POLYBENCH_1D(r,N,n)),
                  int m, int n,
                  DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
                  DATA_TYPE POLYBENCH_1D(s,M,m),
                  DATA_TYPE POLYBENCH_1D(q,N,n),
                  DATA_TYPE POLYBENCH_1D(p,M,m),
                  DATA_TYPE POLYBENCH_1D(r,N,n))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(m, n, A, s, q, p, r);
    
    /* End timer */
    end_time = get_time();
    
    return end_time - start_time;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, M, n, m);
  POLYBENCH_1D_ARRAY_DECL(s, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(q, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);
  
  /* For MKL implementation */
  POLYBENCH_1D_ARRAY_DECL(s_mkl, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(q_mkl, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(r),
	      POLYBENCH_ARRAY(p));

  /* Time naive implementation */
  double naive_time = time_kernel(kernel_bicg, m, n,
                                POLYBENCH_ARRAY(A),
                                POLYBENCH_ARRAY(s),
                                POLYBENCH_ARRAY(q),
                                POLYBENCH_ARRAY(p),
                                POLYBENCH_ARRAY(r));
  
  printf("Naive BICG Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_bicg_mkl, m, n,
                              POLYBENCH_ARRAY(A),
                              POLYBENCH_ARRAY(s_mkl),
                              POLYBENCH_ARRAY(q_mkl),
                              POLYBENCH_ARRAY(p),
                              POLYBENCH_ARRAY(r));
  
  printf("MKL BICG Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(m, n, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(q),
                     POLYBENCH_ARRAY(s_mkl), POLYBENCH_ARRAY(q_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(s);
  POLYBENCH_FREE_ARRAY(q);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(r);
  POLYBENCH_FREE_ARRAY(s_mkl);
  POLYBENCH_FREE_ARRAY(q_mkl);

  return 0;
}
