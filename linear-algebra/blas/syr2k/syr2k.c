/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* syr2k.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "syr2k.h"


/* Array initialization. */
static
void init_array(int n, int m,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		DATA_TYPE POLYBENCH_2D(B,N,M,n,m),
		DATA_TYPE POLYBENCH_2D(C_mkl,N,N,n,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
      A[i][j] = (DATA_TYPE) ((i*j+1)%n) / n;
      B[i][j] = (DATA_TYPE) ((i*j+2)%m) / m;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (DATA_TYPE) ((i*j+3)%n) / m;
      C_mkl[i][j] = C[i][j];
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_syr2k(int n, int m,
		  DATA_TYPE alpha,
		  DATA_TYPE beta,
		  DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
		  DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		  DATA_TYPE POLYBENCH_2D(B,N,M,n,m))
{
  int i, j, k;

//BLAS PARAMS
//UPLO  = 'L'
//TRANS = 'N'
//A is NxM
//B is NxM
//C is NxN
#pragma scop
  for (i = 0; i < _PB_N; i++) {
    for (j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (k = 0; k < _PB_M; k++)
      for (j = 0; j <= i; j++)
	{
	  C[i][j] += A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k];
	}
  }
#pragma endscop

}

/* Intel MKL optimized SYR2K kernel */
static
void kernel_syr2k_mkl(int n, int m,
                    DATA_TYPE alpha,
                    DATA_TYPE beta,
                    DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
                    DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
                    DATA_TYPE POLYBENCH_2D(B,N,M,n,m))
{
#ifdef DATA_TYPE_IS_DOUBLE
  /* Use MKL's double-precision SYR2K: dsyr2k */
  cblas_dsyr2k(CblasRowMajor,     /* Matrix storage order: Row-major (C-style) */
               CblasLower,        /* Which part of matrix C is used: Lower triangular part */
               CblasNoTrans,       /* How to transform matrices A and B: No transpose */
               n,                  /* Number of rows and columns in C */
               m,                  /* Number of columns in A and B */
               alpha,              /* Alpha scalar */
               &A[0][0],           /* Matrix A */
               m,                  /* Leading dimension of A */
               &B[0][0],           /* Matrix B */
               m,                  /* Leading dimension of B */
               beta,               /* Beta scalar */
               &C[0][0],           /* Matrix C */
               n);                 /* Leading dimension of C */
#elif defined(DATA_TYPE_IS_FLOAT)
  /* Use MKL's single-precision SYR2K: ssyr2k */
  cblas_ssyr2k(CblasRowMajor,
               CblasLower,
               CblasNoTrans,
               n, m,
               alpha,
               &A[0][0], m,
               &B[0][0], m,
               beta,
               &C[0][0], n);
#else
  /* For integer types, we fall back to the naive implementation */
  kernel_syr2k(n, m, alpha, beta, C, A, B);
#endif
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int n,
                 DATA_TYPE POLYBENCH_2D(C_naive,N,N,n,n),
                 DATA_TYPE POLYBENCH_2D(C_mkl,N,N,n,n))
{
    int i, j;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    /* Only check the lower triangular part of the matrix, which is what SYR2K updates */
    for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++) {
            diff = fabs(C_naive[i][j] - C_mkl[i][j]);
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
double time_kernel(void (*kernel)(int, int, DATA_TYPE, DATA_TYPE, 
                                 DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
                                 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
                                 DATA_TYPE POLYBENCH_2D(B,N,M,n,m)),
                  int n, int m,
                  DATA_TYPE alpha, DATA_TYPE beta,
                  DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
                  DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
                  DATA_TYPE POLYBENCH_2D(B,N,M,n,m))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(n, m, alpha, beta, C, A, B);
    
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
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,M,n,m);
  
  /* For correctness verification */
  POLYBENCH_2D_ARRAY_DECL(C_mkl,DATA_TYPE,N,N,n,n);

  /* Initialize array(s). */
  init_array (n, m, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
              POLYBENCH_ARRAY(C_mkl));

  /* Time naive implementation */
  double naive_time = time_kernel(kernel_syr2k, n, m, alpha, beta,
                                POLYBENCH_ARRAY(C),
                                POLYBENCH_ARRAY(A),
                                POLYBENCH_ARRAY(B));
  
  printf("Naive SYR2K Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_syr2k_mkl, n, m, alpha, beta,
                              POLYBENCH_ARRAY(C_mkl),
                              POLYBENCH_ARRAY(A),
                              POLYBENCH_ARRAY(B));
  
  printf("MKL SYR2K Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(n, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(C_mkl);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
