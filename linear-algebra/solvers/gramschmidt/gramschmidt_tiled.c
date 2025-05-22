#include <omp.h>
#include <math.h>
#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

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
    DATA_TYPE POLYBENCH_2D(A_ref,M,N,m,n),
    DATA_TYPE POLYBENCH_2D(R_ref,N,N,n,n),
    DATA_TYPE POLYBENCH_2D(Q_ref,M,N,m,n))
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = (((DATA_TYPE) ((i*j) % m) / m )*100) + 10;
      A_ref[i][j] = A[i][j];
      Q[i][j] = 0.0;
      Q_ref[i][j] = 0.0;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      R[i][j] = 0.0;
      R_ref[i][j] = 0.0;
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


static
void kernel_gramschmidt_naive(int m, int n,
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

  int t1, t2, t3, t4, t5, t6, t7, t8, t9;
 register int lbv, ubv;
if (_PB_N >= 1) {
  for (t2=0;t2<=floord(_PB_N-2,32);t2++) {
    for (t4=t2;t4<=floord(_PB_N-1,32);t4++) {
      for (t5=32*t2;t5<=min(min(_PB_N-2,32*t2+31),32*t4+30);t5++) {
        lbv=max(32*t4,t5+1);
        ubv=min(_PB_N-1,32*t4+31);
#pragma ivdep
#pragma vector always
        for (t7=lbv;t7<=ubv;t7++) {
          R[t5][t7] = SCALAR_VAL(0.0);;
        }
      }
    }
  }
  for (t2=0;t2<=_PB_N-1;t2++) {
    nrm = SCALAR_VAL(0.0);;
    for (t4=0;t4<=_PB_M-1;t4++) {
      nrm += A[t4][t2] * A[t4][t2];;
    }
    R[t2][t2] = SQRT_FUN(nrm);;
    for (t4=0;t4<=floord(_PB_M-1,32);t4++) {
      lbv=32*t4;
      ubv=min(_PB_M-1,32*t4+31);
#pragma ivdep
#pragma vector always
      for (t5=lbv;t5<=ubv;t5++) {
        Q[t5][t2] = A[t5][t2] / R[t2][t2];;
      }
    }
    if ((_PB_M >= 1) && (t2 <= _PB_N-2)) {
      for (t4=ceild(t2-30,32);t4<=floord(_PB_N-1,32);t4++) {
        for (t6=0;t6<=floord(_PB_M-1,32);t6++) {
          for (t8=32*t6;t8<=(min(_PB_M-1,32*t6+31))-7;t8+=8) {
            lbv=max(32*t4,t2+1);
            ubv=min(_PB_N-1,32*t4+31);
#pragma ivdep
#pragma vector always
            for (t9=lbv;t9<=ubv;t9++) {
              R[t2][t9] += Q[t8][t2] * A[t8][t9];;
              R[t2][t9] += Q[(t8+1)][t2] * A[(t8+1)][t9];;
              R[t2][t9] += Q[(t8+2)][t2] * A[(t8+2)][t9];;
              R[t2][t9] += Q[(t8+3)][t2] * A[(t8+3)][t9];;
              R[t2][t9] += Q[(t8+4)][t2] * A[(t8+4)][t9];;
              R[t2][t9] += Q[(t8+5)][t2] * A[(t8+5)][t9];;
              R[t2][t9] += Q[(t8+6)][t2] * A[(t8+6)][t9];;
              R[t2][t9] += Q[(t8+7)][t2] * A[(t8+7)][t9];;
            }
          }
          for (;t8<=min(_PB_M-1,32*t6+31);t8++) {
            lbv=max(32*t4,t2+1);
            ubv=min(_PB_N-1,32*t4+31);
#pragma ivdep
#pragma vector always
            for (t9=lbv;t9<=ubv;t9++) {
              R[t2][t9] += Q[t8][t2] * A[t8][t9];;
            }
          }
        }
        for (t6=0;t6<=floord(_PB_M-1,32);t6++) {
          for (t8=32*t6;t8<=(min(_PB_M-1,32*t6+31))-7;t8+=8) {
            lbv=max(32*t4,t2+1);
            ubv=min(_PB_N-1,32*t4+31);
#pragma ivdep
#pragma vector always
            for (t9=lbv;t9<=ubv;t9++) {
              A[t8][t9] = A[t8][t9] - Q[t8][t2] * R[t2][t9];;
              A[(t8+1)][t9] = A[(t8+1)][t9] - Q[(t8+1)][t2] * R[t2][t9];;
              A[(t8+2)][t9] = A[(t8+2)][t9] - Q[(t8+2)][t2] * R[t2][t9];;
              A[(t8+3)][t9] = A[(t8+3)][t9] - Q[(t8+3)][t2] * R[t2][t9];;
              A[(t8+4)][t9] = A[(t8+4)][t9] - Q[(t8+4)][t2] * R[t2][t9];;
              A[(t8+5)][t9] = A[(t8+5)][t9] - Q[(t8+5)][t2] * R[t2][t9];;
              A[(t8+6)][t9] = A[(t8+6)][t9] - Q[(t8+6)][t2] * R[t2][t9];;
              A[(t8+7)][t9] = A[(t8+7)][t9] - Q[(t8+7)][t2] * R[t2][t9];;
            }
          }
          for (;t8<=min(_PB_M-1,32*t6+31);t8++) {
            lbv=max(32*t4,t2+1);
            ubv=min(_PB_N-1,32*t4+31);
#pragma ivdep
#pragma vector always
            for (t9=lbv;t9<=ubv;t9++) {
              A[t8][t9] = A[t8][t9] - Q[t8][t2] * R[t2][t9];;
            }
          }
        }
      }
    }
  }
}

}



/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int m, int n,
                 DATA_TYPE POLYBENCH_2D(R_ref,N,N,n,n),
                 DATA_TYPE POLYBENCH_2D(Q_ref,M,N,m,n),
                 DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
                 DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
    int i, j;
    DATA_TYPE diff_R, diff_Q;
    DATA_TYPE max_diff_R = 0.0;
    DATA_TYPE max_diff_Q = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    /* Check R matrices */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            diff_R = fabs(R_ref[i][j] - R[i][j]);
            if (diff_R > max_diff_R) {
                max_diff_R = diff_R;
            }
        }
    }
    
    /* Check Q matrices */
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            diff_Q = fabs(Q_ref[i][j] - Q[i][j]);
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

  // Reference declaration
  POLYBENCH_2D_ARRAY_DECL(A_ref,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(R_ref,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(Q_ref,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(R),
	      POLYBENCH_ARRAY(Q),
        POLYBENCH_ARRAY(A_ref),
        POLYBENCH_ARRAY(R_ref),
        POLYBENCH_ARRAY(Q_ref));

  /* Time naive implementation */
  double naive_time = time_kernel(kernel_gramschmidt_naive, m, n,
                                POLYBENCH_ARRAY(A_ref),
                                POLYBENCH_ARRAY(R_ref),
                                POLYBENCH_ARRAY(Q_ref));
  
  printf("Naive GRAMSCHMIDT Time: %0.6f seconds\n", naive_time);


  /* Time Pluto implementation */
  double pluto_time = time_kernel(kernel_gramschmidt, m, n,
                                POLYBENCH_ARRAY(A),
                                POLYBENCH_ARRAY(R),
                                POLYBENCH_ARRAY(Q));
  
  printf("Pluto GRAMSCHMIDT Time: %0.6f seconds\n", pluto_time);

  // Speedup
  double speedup = naive_time / pluto_time;
  printf("Speedup: %0.2f\n", speedup);

  /* Verify results */
  verify_results(m, n, POLYBENCH_ARRAY(R_ref), POLYBENCH_ARRAY(Q_ref), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(R);
  POLYBENCH_FREE_ARRAY(Q);

  return 0;
}
