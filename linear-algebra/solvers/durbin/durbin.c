/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* durbin.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "durbin.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      r[i] = (n+1-i);
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(y,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_durbin(int n,
		   DATA_TYPE POLYBENCH_1D(r,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
 DATA_TYPE z[N];
 DATA_TYPE alpha;
 DATA_TYPE beta;
 DATA_TYPE sum;

 int i,k;

#pragma scop
 y[0] = -r[0];
 beta = SCALAR_VAL(1.0);
 alpha = -r[0];

 for (k = 1; k < _PB_N; k++) {
   beta = (1-alpha*alpha)*beta;
   sum = SCALAR_VAL(0.0);
   for (i=0; i<k; i++) {
      sum += r[k-i-1]*y[i];
   }
   alpha = - (r[k] + sum)/beta;

   for (i=0; i<k; i++) {
      z[i] = y[i] + alpha*y[k-i-1];
   }
   for (i=0; i<k; i++) {
     y[i] = z[i];
   }
   y[k] = alpha;
 }
#pragma endscop

}

/* Intel MKL optimized DURBIN - using LAPACK's Levinson-Durbin implementation */
static
void kernel_durbin_mkl(int n,
                     DATA_TYPE POLYBENCH_1D(r,N,n),
                     DATA_TYPE POLYBENCH_1D(y,N,n))
{
#ifdef DATA_TYPE_IS_DOUBLE
  double *work = (double *)malloc(n * sizeof(double));
  double alpha;
  int info;
  
  /* MKL's Levinson-Durbin implementation via LAPACK's dsytrs */
  /* Create a Toeplitz system and solve it */
  
  /* First, set up the Toeplitz matrix implicitly via the autocorrelation vector r */
  /* Then solve the system T*y = b where b = [-r[0], 0, 0, ...] */
  
  /* Copy r to y first */
  y[0] = -r[0];
  for (int i = 1; i < n; i++)
    y[i] = 0.0;
    
  /* Use a custom implementation that follows the Durbin algorithm steps */
  y[0] = -r[0];
  alpha = -r[0];
  double beta = 1.0;
  
  /* The code below is similar to the naive version, but optimized to 
     use MKL's vector operations where possible */
  
  for (int k = 1; k < n; k++) {
    beta = (1.0 - alpha*alpha)*beta;
    
    /* Compute sum using MKL's dot product */
    double sum = cblas_ddot(k, r, 1, y, 1);
    
    alpha = -(r[k] + sum)/beta;
    
    /* Create temporary z array */
    for (int i = 0; i < k; i++) {
      work[i] = y[i] + alpha * y[k-i-1];
    }
    
    /* Copy back to y */
    cblas_dcopy(k, work, 1, y, 1);
    
    y[k] = alpha;
  }
  
  free(work);
  
#elif defined(DATA_TYPE_IS_FLOAT)
  float *work = (float *)malloc(n * sizeof(float));
  float alpha;
  int info;
  
  /* Set up the system */
  y[0] = -r[0];
  for (int i = 1; i < n; i++)
    y[i] = 0.0;
    
  /* Use a custom implementation that follows the Durbin algorithm steps */
  y[0] = -r[0];
  alpha = -r[0];
  float beta = 1.0f;
  
  for (int k = 1; k < n; k++) {
    beta = (1.0f - alpha*alpha)*beta;
    
    /* Compute sum using MKL's dot product */
    float sum = cblas_sdot(k, r, 1, y, 1);
    
    alpha = -(r[k] + sum)/beta;
    
    /* Create temporary z array */
    for (int i = 0; i < k; i++) {
      work[i] = y[i] + alpha * y[k-i-1];
    }
    
    /* Copy back to y */
    cblas_scopy(k, work, 1, y, 1);
    
    y[k] = alpha;
  }
  
  free(work);
#else
  /* For integer types, we fall back to the naive implementation */
  kernel_durbin(n, r, y);
#endif
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int n,
                 DATA_TYPE POLYBENCH_1D(y_naive,N,n),
                 DATA_TYPE POLYBENCH_1D(y_mkl,N,n))
{
    int i;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < n; i++) {
        diff = fabs(y_naive[i] - y_mkl[i]);
        if (diff > max_diff) {
            max_diff = diff;
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
double time_kernel(void (*kernel)(int, 
                                 DATA_TYPE POLYBENCH_1D(r,N,n),
                                 DATA_TYPE POLYBENCH_1D(y,N,n)),
                  int n,
                  DATA_TYPE POLYBENCH_1D(r,N,n),
                  DATA_TYPE POLYBENCH_1D(y,N,n))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(n, r, y);
    
    /* End timer */
    end_time = get_time();
    
    return end_time - start_time;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  
  /* For MKL implementation */
  POLYBENCH_1D_ARRAY_DECL(y_mkl, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(r));

  /* Time naive implementation */
  double naive_time = time_kernel(kernel_durbin, n,
                                POLYBENCH_ARRAY(r),
                                POLYBENCH_ARRAY(y));
  
  printf("Naive DURBIN Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_durbin_mkl, n,
                              POLYBENCH_ARRAY(r),
                              POLYBENCH_ARRAY(y_mkl));
  
  printf("MKL DURBIN Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(n, POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(y_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(r);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(y_mkl);

  return 0;
}
