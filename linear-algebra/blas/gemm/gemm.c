/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
    DATA_TYPE POLYBENCH_2D(C_mkl,NI,NJ,ni,nj))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % nk) / nk;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+2) % nj) / nj;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C_mkl[i][j] = C[i][j];
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemm(int ni, int nj, int nk,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		 DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j, k;

//BLAS PARAMS
//TRANSA = 'N'
//TRANSB = 'N'
// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
#pragma scop
  for (i = 0; i < _PB_NI; i++) {
    for (j = 0; j < _PB_NJ; j++)
	C[i][j] *= beta;
    for (k = 0; k < _PB_NK; k++) {
       for (j = 0; j < _PB_NJ; j++)
	  C[i][j] += alpha * A[i][k] * B[k][j];
    }
  }
#pragma endscop

}

/* Intel MKL optimized GEMM kernel */
static
void kernel_gemm_mkl(int ni, int nj, int nk,
                    DATA_TYPE alpha,
                    DATA_TYPE beta,
                    DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
                    DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                    DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
#ifdef DATA_TYPE_IS_DOUBLE
    /* Use MKL's double-precision GEMM: dgemm */
    cblas_dgemm(CblasRowMajor,     /* Matrix storage order: Row-major (C-style) */
                CblasNoTrans,       /* A: no transpose */
                CblasNoTrans,       /* B: no transpose */
                ni, nj, nk,         /* Matrix dimensions */
                alpha,              /* Alpha scalar */
                &A[0][0], nk,       /* Matrix A and leading dimension */
                &B[0][0], nj,       /* Matrix B and leading dimension */
                beta,               /* Beta scalar */
                &C[0][0], nj);      /* Matrix C and leading dimension */
#elif defined(DATA_TYPE_IS_FLOAT)
    /* Use MKL's single-precision GEMM: sgemm */
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                ni, nj, nk,
                alpha,
                &A[0][0], nk,
                &B[0][0], nj,
                beta,
                &C[0][0], nj);
#else
    /* For integer types, we could potentially use MKL's integer GEMM (if available) 
       or manually implement the GEMM operation with MKL vectorization primitives.
       For simplicity, we'll fall back to the naive implementation for integer types. */
    kernel_gemm(ni, nj, nk, alpha, beta, C, A, B);
#endif
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int ni, int nj,
                 DATA_TYPE POLYBENCH_2D(C_naive,NI,NJ,ni,nj),
                 DATA_TYPE POLYBENCH_2D(C_mkl,NI,NJ,ni,nj))
{
    int i, j;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-6;
    
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
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
double time_kernel(void (*kernel)(int, int, int, DATA_TYPE, DATA_TYPE, 
                                 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
                                 DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                                 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj)),
                  int ni, int nj, int nk,
                  DATA_TYPE alpha, DATA_TYPE beta,
                  DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
                  DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                  DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(ni, nj, nk, alpha, beta, C, A, B);
    
    /* End timer */
    end_time = get_time();
    
    return end_time - start_time;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
  
  /* For correctness verification */
  POLYBENCH_2D_ARRAY_DECL(C_mkl,DATA_TYPE,NI,NJ,ni,nj);

  /* Initialize array(s). */
  init_array (ni, nj, nk, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C_mkl));
  
  // /* Make a copy of C for MKL implementation */
  // int i, j;
  // for (i = 0; i < ni; i++) {
  //   for (j = 0; j < nj; j++) {
  //     C_mkl[i][j] = C[i][j];
  //   }
  // }

  /* Start timer for naive implementation. */
  double naive_time = time_kernel(kernel_gemm, ni, nj, nk, alpha, beta,
                                POLYBENCH_ARRAY(C),
                                POLYBENCH_ARRAY(A),
                                POLYBENCH_ARRAY(B));
  
  printf("Naive GEMM Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_gemm_mkl, ni, nj, nk, alpha, beta,
                              POLYBENCH_ARRAY(C_mkl),
                              POLYBENCH_ARRAY(A),
                              POLYBENCH_ARRAY(B));
  
  printf("MKL GEMM Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(ni, nj, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(C_mkl);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
