/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* symm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "symm.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		DATA_TYPE POLYBENCH_2D(B,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(C_mkl,M,N,m,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
      C_mkl[i][j] = C[i][j];
      B[i][j] = (DATA_TYPE) ((n+i-j) % 100) / m;
    }
  for (i = 0; i < m; i++) {
    for (j = 0; j <=i; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
    for (j = i+1; j < m; j++)
      A[i][j] = -999; //regions of arrays that should not be used
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(C,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_symm(int m, int n,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
		 DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j, k;
  DATA_TYPE temp2;

//BLAS PARAMS
//SIDE = 'L'
//UPLO = 'L'
// =>  Form  C := alpha*A*B + beta*C
// A is MxM
// B is MxN
// C is MxN
//note that due to Fortran array layout, the code below more closely resembles upper triangular case in BLAS
#pragma scop
   for (i = 0; i < _PB_M; i++)
      for (j = 0; j < _PB_N; j++ )
      {
        temp2 = 0;
        for (k = 0; k < i; k++) {
           C[k][j] += alpha*B[i][j] * A[i][k];
           temp2 += B[k][j] * A[i][k];
        }
        C[i][j] = beta * C[i][j] + alpha*B[i][j] * A[i][i] + alpha * temp2;
     }
#pragma endscop

}

/* Intel MKL optimized SYMM kernel */
static
void kernel_symm_mkl(int m, int n,
                   DATA_TYPE alpha,
                   DATA_TYPE beta,
                   DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
                   DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
                   DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
#ifdef DATA_TYPE_IS_DOUBLE
  /* Use MKL's double-precision SYMM: dsymm */
  cblas_dsymm(CblasRowMajor,      /* Matrix storage order: Row-major (C-style) */
              CblasLeft,          /* Side: A is to the left of B */
              CblasLower,         /* Triangle: Lower triangular part of A is valid */
              m, n,               /* Matrix dimensions */
              alpha,              /* Alpha scalar */
              &A[0][0], m,        /* Matrix A and leading dimension */
              &B[0][0], n,        /* Matrix B and leading dimension */
              beta,               /* Beta scalar */
              &C[0][0], n);       /* Matrix C and leading dimension */
#elif defined(DATA_TYPE_IS_FLOAT)
  /* Use MKL's single-precision SYMM: ssymm */
  cblas_ssymm(CblasRowMajor,
              CblasLeft,
              CblasLower,
              m, n,
              alpha,
              &A[0][0], m,
              &B[0][0], n,
              beta,
              &C[0][0], n);
#else
  /* For integer types, we fall back to the naive implementation */
  kernel_symm(m, n, alpha, beta, C, A, B);
#endif
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int m, int n,
                 DATA_TYPE POLYBENCH_2D(C_naive,M,N,m,n),
                 DATA_TYPE POLYBENCH_2D(C_mkl,M,N,m,n))
{
    int i, j;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
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
                                 DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
                                 DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
                                 DATA_TYPE POLYBENCH_2D(B,M,N,m,n)),
                  int m, int n,
                  DATA_TYPE alpha, DATA_TYPE beta,
                  DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
                  DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
                  DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(m, n, alpha, beta, C, A, B);
    
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
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,M,m,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,M,N,m,n);
  
  /* For correctness verification */
  POLYBENCH_2D_ARRAY_DECL(C_mkl,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C_mkl));

  /* Time naive implementation */
  double naive_time = time_kernel(kernel_symm, m, n, alpha, beta,
                                POLYBENCH_ARRAY(C),
                                POLYBENCH_ARRAY(A),
                                POLYBENCH_ARRAY(B));
  
  printf("Naive SYMM Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_symm_mkl, m, n, alpha, beta,
                              POLYBENCH_ARRAY(C_mkl),
                              POLYBENCH_ARRAY(A),
                              POLYBENCH_ARRAY(B));
  
  printf("MKL SYMM Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(m, n, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(C_mkl);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
