/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* doitgen.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "doitgen.h"


/* Array initialization. */
static
void init_array(int nr, int nq, int np,
		DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		DATA_TYPE POLYBENCH_3D(A_mkl,NR,NQ,NP,nr,nq,np))
{
  int i, j, k;

  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++) {
	A[i][j][k] = (DATA_TYPE) ((i*j + k)%np) / np;
	A_mkl[i][j][k] = A[i][j][k];
      }
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++)
      C4[i][j] = (DATA_TYPE) (i*j % np) / np;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nr, int nq, int np,
		 DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np))
{
  int i, j, k;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++) {
	if ((i*nq*np+j*np+k) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
      }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_doitgen(int nr, int nq, int np,
		    DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_1D(sum,NP,np))
{
  int r, q, p, s;

#pragma scop
  for (r = 0; r < _PB_NR; r++)
    for (q = 0; q < _PB_NQ; q++)  {
      for (p = 0; p < _PB_NP; p++)  {
	sum[p] = SCALAR_VAL(0.0);
	for (s = 0; s < _PB_NP; s++)
	  sum[p] += A[r][q][s] * C4[s][p];
      }
      for (p = 0; p < _PB_NP; p++)
	A[r][q][p] = sum[p];
    }
#pragma endscop

}

/* Intel MKL optimized DOITGEN kernel using GEMM for the matrix multiplication */
void kernel_doitgen_mkl(int nr, int nq, int np,
                      DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
                      DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
                      DATA_TYPE POLYBENCH_1D(sum,NP,np))
{
  int r, q;

#ifdef DATA_TYPE_IS_DOUBLE
  const double alpha = 1.0;
  const double beta = 0.0;
  
  /* For each r, q, compute sum = A[r,q,:] * C4 using matrix-matrix multiplication */
  for (r = 0; r < nr; r++) {
    for (q = 0; q < nq; q++) {
      /* Use GEMV (matrix-vector multiplication) from MKL 
         A[r][q][:] is treated as a row vector, C4 is the matrix, and sum is the result vector
         sum = A[r][q][:] * C4 */
      cblas_dgemv(CblasRowMajor,     /* Matrix storage order: Row-major (C-style) */
                  CblasNoTrans,      /* No transpose of matrix */
                  np, np,            /* Dimensions of C4 */
                  alpha,             /* Alpha scalar */
                  &C4[0][0], np,     /* C4 matrix and leading dimension */
                  &A[r][q][0], 1,    /* A[r][q][:] vector and stride */
                  beta,              /* Beta scalar */
                  sum, 1);           /* Output vector and stride */

      /* Copy result back to A[r][q][:] */
      memcpy(&A[r][q][0], sum, np * sizeof(DATA_TYPE));
    }
  }
  
#elif defined(DATA_TYPE_IS_FLOAT)
  const float alpha = 1.0f;
  const float beta = 0.0f;
  
  /* For each r, q, compute sum = A[r,q,:] * C4 using matrix-vector multiplication */
  for (r = 0; r < nr; r++) {
    for (q = 0; q < nq; q++) {
      /* Use GEMV (matrix-vector multiplication) from MKL 
         A[r][q][:] is treated as a row vector, C4 is the matrix, and sum is the result vector
         sum = A[r][q][:] * C4 */
      cblas_sgemv(CblasRowMajor,     /* Matrix storage order: Row-major (C-style) */
                  CblasNoTrans,      /* No transpose of matrix */
                  np, np,            /* Dimensions of C4 */
                  alpha,             /* Alpha scalar */
                  &C4[0][0], np,     /* C4 matrix and leading dimension */
                  &A[r][q][0], 1,    /* A[r][q][:] vector and stride */
                  beta,              /* Beta scalar */
                  sum, 1);           /* Output vector and stride */

      /* Copy result back to A[r][q][:] */
      memcpy(&A[r][q][0], sum, np * sizeof(DATA_TYPE));
    }
  }
#else
  /* For integer types, we fall back to the naive implementation */
  kernel_doitgen(nr, nq, np, A, C4, sum);
#endif
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int nr, int nq, int np,
                 DATA_TYPE POLYBENCH_3D(A_naive,NR,NQ,NP,nr,nq,np),
                 DATA_TYPE POLYBENCH_3D(A_mkl,NR,NQ,NP,nr,nq,np))
{
    int i, j, k;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < nr; i++) {
        for (j = 0; j < nq; j++) {
            for (k = 0; k < np; k++) {
                diff = fabs(A_naive[i][j][k] - A_mkl[i][j][k]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
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
double time_kernel(void (*kernel)(int, int, int,
                                 DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
                                 DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
                                 DATA_TYPE POLYBENCH_1D(sum,NP,np)),
                  int nr, int nq, int np,
                  DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
                  DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
                  DATA_TYPE POLYBENCH_1D(sum,NP,np))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(nr, nq, np, A, C4, sum);
    
    /* End timer */
    end_time = get_time();
    
    return end_time - start_time;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int nr = NR;
  int nq = NQ;
  int np = NP;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A,DATA_TYPE,NR,NQ,NP,nr,nq,np);
  POLYBENCH_1D_ARRAY_DECL(sum,DATA_TYPE,NP,np);
  POLYBENCH_2D_ARRAY_DECL(C4,DATA_TYPE,NP,NP,np,np);
  
  /* For MKL implementation */
  POLYBENCH_3D_ARRAY_DECL(A_mkl,DATA_TYPE,NR,NQ,NP,nr,nq,np);
  POLYBENCH_1D_ARRAY_DECL(sum_mkl,DATA_TYPE,NP,np);

  /* Initialize array(s). */
  init_array (nr, nq, np,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(C4),
	      POLYBENCH_ARRAY(A_mkl));

  /* Time naive implementation */
  double naive_time = time_kernel(kernel_doitgen, nr, nq, np,
                                POLYBENCH_ARRAY(A),
                                POLYBENCH_ARRAY(C4),
                                POLYBENCH_ARRAY(sum));
  
  printf("Naive DOITGEN Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_doitgen_mkl, nr, nq, np,
                              POLYBENCH_ARRAY(A_mkl),
                              POLYBENCH_ARRAY(C4),
                              POLYBENCH_ARRAY(sum_mkl));
  
  printf("MKL DOITGEN Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(nr, nq, np, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nr, nq, np, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(A_mkl);
  POLYBENCH_FREE_ARRAY(sum);
  POLYBENCH_FREE_ARRAY(sum_mkl);
  POLYBENCH_FREE_ARRAY(C4);

  return 0;
}
