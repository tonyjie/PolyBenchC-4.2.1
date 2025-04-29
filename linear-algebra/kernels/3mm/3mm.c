/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* 3mm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "3mm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1) % ni) / (5*ni);
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) ((i*(j+1)+2) % nj) / (5*nj);
  for (i = 0; i < nj; i++)
    for (j = 0; j < nm; j++)
      C[i][j] = (DATA_TYPE) (i*(j+3) % nl) / (5*nl);
  for (i = 0; i < nm; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (DATA_TYPE) ((i*(j+2)+2) % nk) / (5*nk);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("G");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, G[i][j]);
    }
  POLYBENCH_DUMP_END("G");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_3mm(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
		DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j, k;

#pragma scop
  /* E := A*B */
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NJ; j++)
      {
	E[i][j] = SCALAR_VAL(0.0);
	for (k = 0; k < _PB_NK; ++k)
	  E[i][j] += A[i][k] * B[k][j];
      }
  /* F := C*D */
  for (i = 0; i < _PB_NJ; i++)
    for (j = 0; j < _PB_NL; j++)
      {
	F[i][j] = SCALAR_VAL(0.0);
	for (k = 0; k < _PB_NM; ++k)
	  F[i][j] += C[i][k] * D[k][j];
      }
  /* G := E*F */
  for (i = 0; i < _PB_NI; i++)
    for (j = 0; j < _PB_NL; j++)
      {
	G[i][j] = SCALAR_VAL(0.0);
	for (k = 0; k < _PB_NJ; ++k)
	  G[i][j] += E[i][k] * F[k][j];
      }
#pragma endscop

}

/* Intel MKL optimized 3MM kernel */
static
void kernel_3mm_mkl(int ni, int nj, int nk, int nl, int nm,
                  DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
                  DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                  DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                  DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
                  DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
                  DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
                  DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
#ifdef DATA_TYPE_IS_DOUBLE
  const double alpha = 1.0;
  const double beta = 0.0;

  /* E := alpha*A*B + beta*E */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
              ni, nj, nk, 
              alpha, &A[0][0], nk, &B[0][0], nj, 
              beta, &E[0][0], nj);
              
  /* F := alpha*C*D + beta*F */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
              nj, nl, nm, 
              alpha, &C[0][0], nm, &D[0][0], nl, 
              beta, &F[0][0], nl);
              
  /* G := alpha*E*F + beta*G */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
              ni, nl, nj, 
              alpha, &E[0][0], nj, &F[0][0], nl, 
              beta, &G[0][0], nl);

#elif defined(DATA_TYPE_IS_FLOAT)
  const float alpha = 1.0f;
  const float beta = 0.0f;

  /* E := alpha*A*B + beta*E */
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
              ni, nj, nk, 
              alpha, &A[0][0], nk, &B[0][0], nj, 
              beta, &E[0][0], nj);
              
  /* F := alpha*C*D + beta*F */
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
              nj, nl, nm, 
              alpha, &C[0][0], nm, &D[0][0], nl, 
              beta, &F[0][0], nl);
              
  /* G := alpha*E*F + beta*G */
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
              ni, nl, nj, 
              alpha, &E[0][0], nj, &F[0][0], nl, 
              beta, &G[0][0], nl);
#else
  /* For integer types, we fall back to the naive implementation */
  kernel_3mm(ni, nj, nk, nl, nm, E, A, B, F, C, D, G);
#endif
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int ni, int nl,
                 DATA_TYPE POLYBENCH_2D(G_naive,NI,NL,ni,nl),
                 DATA_TYPE POLYBENCH_2D(G_mkl,NI,NL,ni,nl))
{
    int i, j;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nl; j++) {
            diff = fabs(G_naive[i][j] - G_mkl[i][j]);
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
double time_kernel(void (*kernel)(int, int, int, int, int,
                                 DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
                                 DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                                 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                                 DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
                                 DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
                                 DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
                                 DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl)),
                  int ni, int nj, int nk, int nl, int nm,
                  DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
                  DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                  DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                  DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
                  DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
                  DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
                  DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(ni, nj, nk, nl, nm, E, A, B, F, C, D, G);
    
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
  int nm = NM;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
  POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
  POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
  POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);
  
  /* For MKL implementation */
  POLYBENCH_2D_ARRAY_DECL(E_mkl, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(F_mkl, DATA_TYPE, NJ, NL, nj, nl);
  POLYBENCH_2D_ARRAY_DECL(G_mkl, DATA_TYPE, NI, NL, ni, nl);

  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, nm,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));

  /* Time naive implementation */
  double naive_time = time_kernel(kernel_3mm, ni, nj, nk, nl, nm,
                                POLYBENCH_ARRAY(E),
                                POLYBENCH_ARRAY(A),
                                POLYBENCH_ARRAY(B),
                                POLYBENCH_ARRAY(F),
                                POLYBENCH_ARRAY(C),
                                POLYBENCH_ARRAY(D),
                                POLYBENCH_ARRAY(G));
  
  printf("Naive 3MM Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_3mm_mkl, ni, nj, nk, nl, nm,
                              POLYBENCH_ARRAY(E_mkl),
                              POLYBENCH_ARRAY(A),
                              POLYBENCH_ARRAY(B),
                              POLYBENCH_ARRAY(F_mkl),
                              POLYBENCH_ARRAY(C),
                              POLYBENCH_ARRAY(D),
                              POLYBENCH_ARRAY(G_mkl));
  
  printf("MKL 3MM Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(ni, nl, POLYBENCH_ARRAY(G), POLYBENCH_ARRAY(G_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl, POLYBENCH_ARRAY(G)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(E);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(F);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);
  POLYBENCH_FREE_ARRAY(G);
  POLYBENCH_FREE_ARRAY(E_mkl);
  POLYBENCH_FREE_ARRAY(F_mkl);
  POLYBENCH_FREE_ARRAY(G_mkl);

  return 0;
}
