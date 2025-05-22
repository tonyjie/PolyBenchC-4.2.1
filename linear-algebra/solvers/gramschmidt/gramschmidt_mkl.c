/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gramschmidt.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gramschmidt.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(Q,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(A_mkl,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(R_mkl,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(Q_mkl,M,N,m,n))
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = (((DATA_TYPE) ((i*j) % m) / m )*100) + 10;
      A_mkl[i][j] = A[i][j];
      Q[i][j] = 0.0;
      Q_mkl[i][j] = 0.0;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      R[i][j] = 0.0;
      R_mkl[i][j] = 0.0;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("R");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, R[i][j]);
    }
  POLYBENCH_DUMP_END("R");

  POLYBENCH_DUMP_BEGIN("Q");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, Q[i][j]);
    }
  POLYBENCH_DUMP_END("Q");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* QR Decomposition with Modified Gram Schmidt:
 http://www.inf.ethz.ch/personal/gander/ */
static
void kernel_gramschmidt(int m, int n,
			DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
			DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
			DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j, k;

  DATA_TYPE nrm;

#pragma scop
  for (k = 0; k < _PB_N; k++)
    {
      nrm = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_M; i++)
        nrm += A[i][k] * A[i][k];
      R[k][k] = SQRT_FUN(nrm);
      for (i = 0; i < _PB_M; i++)
        Q[i][k] = A[i][k] / R[k][k];
      for (j = k + 1; j < _PB_N; j++)
	{
	  R[k][j] = SCALAR_VAL(0.0);
	  for (i = 0; i < _PB_M; i++)
	    R[k][j] += Q[i][k] * A[i][j];
	  for (i = 0; i < _PB_M; i++)
	    A[i][j] = A[i][j] - Q[i][k] * R[k][j];
	}
    }
#pragma endscop

}

/* Intel MKL optimized GRAMSCHMIDT using LAPACK's QR factorization */
static
void kernel_gramschmidt_mkl(int m, int n,
                         DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
                         DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
                         DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
#ifdef DATA_TYPE_IS_DOUBLE
  /* The LAPACK QR implementation (dgeqrf + dorgqr) doesn't apply exactly 
     the same algorithm as the naive implementation, so we implement the algorithm
     directly using BLAS operations for optimal performance */
  
  double nrm;
  double *work = (double *)malloc(m * sizeof(double));
  
  /* Copy A to Q initially, since we'll need to preserve A */
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      Q[i][j] = A[i][j];
      
  for (int k = 0; k < n; k++) {
    /* Calculate norm of column vector: nrm = ||Q(:,k)||_2 */
    nrm = cblas_dnrm2(m, &Q[0][k], n);
    R[k][k] = nrm;
    
    /* Q(:,k) = Q(:,k) / R(k,k) */
    cblas_dscal(m, 1.0/nrm, &Q[0][k], n);
    
    /* Update the remaining columns */
    for (int j = k + 1; j < n; j++) {
      /* R(k,j) = Q(:,k)' * A(:,j) */
      R[k][j] = cblas_ddot(m, &Q[0][k], n, &A[0][j], n);
      
      /* A(:,j) = A(:,j) - Q(:,k) * R(k,j) */
      cblas_daxpy(m, -R[k][j], &Q[0][k], n, &A[0][j], n);
    }
  }
  
  free(work);
  
#elif defined(DATA_TYPE_IS_FLOAT)
  float nrm;
  float *work = (float *)malloc(m * sizeof(float));
  
  /* Copy A to Q initially */
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      Q[i][j] = A[i][j];
      
  for (int k = 0; k < n; k++) {
    /* Calculate norm of column vector: nrm = ||Q(:,k)||_2 */
    nrm = cblas_snrm2(m, &Q[0][k], n);
    R[k][k] = nrm;
    
    /* Q(:,k) = Q(:,k) / R(k,k) */
    cblas_sscal(m, 1.0f/nrm, &Q[0][k], n);
    
    /* Update the remaining columns */
    for (int j = k + 1; j < n; j++) {
      /* R(k,j) = Q(:,k)' * A(:,j) */
      R[k][j] = cblas_sdot(m, &Q[0][k], n, &A[0][j], n);
      
      /* A(:,j) = A(:,j) - Q(:,k) * R(k,j) */
      cblas_saxpy(m, -R[k][j], &Q[0][k], n, &A[0][j], n);
    }
  }
  
  free(work);
#else
  /* For integer types, we fall back to the naive implementation */
  kernel_gramschmidt(m, n, A, R, Q);
#endif
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int m, int n,
                 DATA_TYPE POLYBENCH_2D(R_naive,N,N,n,n),
                 DATA_TYPE POLYBENCH_2D(Q_naive,M,N,m,n),
                 DATA_TYPE POLYBENCH_2D(R_mkl,N,N,n,n),
                 DATA_TYPE POLYBENCH_2D(Q_mkl,M,N,m,n))
{
    int i, j;
    DATA_TYPE diff_R, diff_Q;
    DATA_TYPE max_diff_R = 0.0;
    DATA_TYPE max_diff_Q = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    /* Check R matrices */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            diff_R = fabs(R_naive[i][j] - R_mkl[i][j]);
            if (diff_R > max_diff_R) {
                max_diff_R = diff_R;
            }
        }
    }
    
    /* Check Q matrices */
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            diff_Q = fabs(Q_naive[i][j] - Q_mkl[i][j]);
            if (diff_Q > max_diff_Q) {
                max_diff_Q = diff_Q;
            }
        }
    }
    
    printf("Maximum difference in R matrices: %e\n", max_diff_R);
    printf("Maximum difference in Q matrices: %e\n", max_diff_Q);
    
    if (max_diff_R < threshold && max_diff_Q < threshold) {
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
                                 DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
                                 DATA_TYPE POLYBENCH_2D(Q,M,N,m,n)),
                  int m, int n,
                  DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
                  DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
                  DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(m, n, A, R, Q);
    
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
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(R,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(Q,DATA_TYPE,M,N,m,n);
  
  /* For MKL implementation */
  POLYBENCH_2D_ARRAY_DECL(A_mkl,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(R_mkl,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(Q_mkl,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(R),
	      POLYBENCH_ARRAY(Q),
	      POLYBENCH_ARRAY(A_mkl),
	      POLYBENCH_ARRAY(R_mkl),
	      POLYBENCH_ARRAY(Q_mkl));

  /* Time naive implementation */
  double naive_time = time_kernel(kernel_gramschmidt, m, n,
                                POLYBENCH_ARRAY(A),
                                POLYBENCH_ARRAY(R),
                                POLYBENCH_ARRAY(Q));
  
  printf("Naive GRAMSCHMIDT Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_gramschmidt_mkl, m, n,
                              POLYBENCH_ARRAY(A_mkl),
                              POLYBENCH_ARRAY(R_mkl),
                              POLYBENCH_ARRAY(Q_mkl));
  
  printf("MKL GRAMSCHMIDT Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(m, n, 
                POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q),
                POLYBENCH_ARRAY(R_mkl), POLYBENCH_ARRAY(Q_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(R);
  POLYBENCH_FREE_ARRAY(Q);
  POLYBENCH_FREE_ARRAY(A_mkl);
  POLYBENCH_FREE_ARRAY(R_mkl);
  POLYBENCH_FREE_ARRAY(Q_mkl);

  return 0;
}
