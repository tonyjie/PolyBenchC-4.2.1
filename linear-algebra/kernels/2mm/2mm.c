/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* 2mm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "2mm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl),
		DATA_TYPE POLYBENCH_2D(D_mkl,NI,NL,ni,nl))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+1) % nj) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      C[i][j] = (DATA_TYPE) ((i*(j+3)+1) % nl) / nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
      D[i][j] = (DATA_TYPE) (i*(j+2) % nk) / nk;
      D_mkl[i][j] = D[i][j];
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("D");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, D[i][j]);
    }
  POLYBENCH_DUMP_END("D");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_2mm(int ni, int nj, int nk, int nl,
		DATA_TYPE alpha,
		DATA_TYPE beta,
		DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j, k;

#pragma scop
  /* D := alpha*A*B*C + beta*D */
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NJ; j++)
      {
	tmp[i][j] = SCALAR_VAL(0.0);
	for (k = 0; k < _PB_NK; ++k)
	  tmp[i][j] += alpha * A[i][k] * B[k][j];
      }
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NL; j++)
      {
	D[i][j] *= beta;
	for (k = 0; k < _PB_NJ; ++k)
	  D[i][j] += tmp[i][k] * C[k][j];
      }
#pragma endscop

}

/* Intel MKL optimized 2MM kernel */
static
void kernel_2mm_mkl(int ni, int nj, int nk, int nl,
                   DATA_TYPE alpha,
                   DATA_TYPE beta,
                   DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
                   DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                   DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                   DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
                   DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
#ifdef DATA_TYPE_IS_DOUBLE
  /* tmp = alpha*A*B, using MKL's dgemm */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
              ni, nj, nk, 
              alpha, &A[0][0], nk, &B[0][0], nj, 
              0.0, &tmp[0][0], nj);
  
  /* Scale D by beta */
  for (int i = 0; i < ni; i++) {
    cblas_dscal(nl, beta, &D[i][0], 1);
  }
  
  /* D = tmp*C + D, using MKL's dgemm */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
              ni, nl, nj, 
              1.0, &tmp[0][0], nj, &C[0][0], nl, 
              1.0, &D[0][0], nl);
              
#elif defined(DATA_TYPE_IS_FLOAT)
  /* tmp = alpha*A*B, using MKL's sgemm */
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
              ni, nj, nk, 
              alpha, &A[0][0], nk, &B[0][0], nj, 
              0.0f, &tmp[0][0], nj);
  
  /* Scale D by beta */
  for (int i = 0; i < ni; i++) {
    cblas_sscal(nl, beta, &D[i][0], 1);
  }
  
  /* D = tmp*C + D, using MKL's sgemm */
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
              ni, nl, nj, 
              1.0f, &tmp[0][0], nj, &C[0][0], nl, 
              1.0f, &D[0][0], nl);
#else
  /* For integer types, fall back to the naive implementation */
  kernel_2mm(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D);
#endif
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int ni, int nl,
                 DATA_TYPE POLYBENCH_2D(D_naive,NI,NL,ni,nl),
                 DATA_TYPE POLYBENCH_2D(D_mkl,NI,NL,ni,nl))
{
    int i, j;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nl; j++) {
            diff = fabs(D_naive[i][j] - D_mkl[i][j]);
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
double time_kernel(void (*kernel)(int, int, int, int, DATA_TYPE, DATA_TYPE, 
                                 DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
                                 DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                                 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                                 DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
                                 DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl)),
                  int ni, int nj, int nk, int nl,
                  DATA_TYPE alpha, DATA_TYPE beta,
                  DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
                  DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                  DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                  DATA_TYPE POLYBENCH_2D(C,NJ,NL,nj,nl),
                  DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D);
    
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
  int nl = NL;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NJ,NL,nj,nl);
  POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);
  
  /* For MKL implementation */
  POLYBENCH_2D_ARRAY_DECL(tmp_mkl,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(D_mkl,DATA_TYPE,NI,NL,ni,nl);

  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D),
	      POLYBENCH_ARRAY(D_mkl));

  /* Time naive implementation */
  double naive_time = time_kernel(kernel_2mm, ni, nj, nk, nl, alpha, beta,
                                POLYBENCH_ARRAY(tmp),
                                POLYBENCH_ARRAY(A),
                                POLYBENCH_ARRAY(B),
                                POLYBENCH_ARRAY(C),
                                POLYBENCH_ARRAY(D));
  
  printf("Naive 2MM Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_2mm_mkl, ni, nj, nk, nl, alpha, beta,
                              POLYBENCH_ARRAY(tmp_mkl),
                              POLYBENCH_ARRAY(A),
                              POLYBENCH_ARRAY(B),
                              POLYBENCH_ARRAY(C),
                              POLYBENCH_ARRAY(D_mkl));
  
  printf("MKL 2MM Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(ni, nl, POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(D_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl, POLYBENCH_ARRAY(D)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(tmp_mkl);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);
  POLYBENCH_FREE_ARRAY(D_mkl);

  return 0;
}
