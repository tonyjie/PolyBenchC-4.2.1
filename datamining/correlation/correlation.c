/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* correlation.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "correlation.h"


/* Array initialization. */
static
void init_array (int m,
		 int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
		 DATA_TYPE POLYBENCH_2D(data_mkl,N,M,n,m))
{
  int i, j;

  *float_n = (DATA_TYPE)N;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++) {
      data[i][j] = (DATA_TYPE)(i*j)/M + i;
      data_mkl[i][j] = data[i][j];
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(corr,M,M,m,m))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("corr");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, corr[i][j]);
    }
  POLYBENCH_DUMP_END("corr");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_correlation(int m, int n,
			DATA_TYPE float_n,
			DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
			DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
			DATA_TYPE POLYBENCH_1D(mean,M,m),
			DATA_TYPE POLYBENCH_1D(stddev,M,m))
{
  int i, j, k;

  DATA_TYPE eps = SCALAR_VAL(0.1);


#pragma scop
  for (j = 0; j < _PB_M; j++)
    {
      mean[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
	mean[j] += data[i][j];
      mean[j] /= float_n;
    }


   for (j = 0; j < _PB_M; j++)
    {
      stddev[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
        stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      stddev[j] /= float_n;
      stddev[j] = SQRT_FUN(stddev[j]);
      /* The following in an inelegant but usual way to handle
         near-zero std. dev. values, which below would cause a zero-
         divide. */
      stddev[j] = stddev[j] <= eps ? SCALAR_VAL(1.0) : stddev[j];
    }

  /* Center and reduce the column vectors. */
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      {
        data[i][j] -= mean[j];
        data[i][j] /= SQRT_FUN(float_n) * stddev[j];
      }

  /* Calculate the m * m correlation matrix. */
  for (i = 0; i < _PB_M-1; i++)
    {
      corr[i][i] = SCALAR_VAL(1.0);
      for (j = i+1; j < _PB_M; j++)
        {
          corr[i][j] = SCALAR_VAL(0.0);
          for (k = 0; k < _PB_N; k++)
            corr[i][j] += (data[k][i] * data[k][j]);
          corr[j][i] = corr[i][j];
        }
    }
  corr[_PB_M-1][_PB_M-1] = SCALAR_VAL(1.0);
#pragma endscop
}

/* Intel MKL optimized correlation kernel */
static
void kernel_correlation_mkl(int m, int n,
                          DATA_TYPE float_n,
                          DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
                          DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
                          DATA_TYPE POLYBENCH_1D(mean,M,m),
                          DATA_TYPE POLYBENCH_1D(stddev,M,m))
{
  int i, j;
  DATA_TYPE eps = SCALAR_VAL(0.1);
  DATA_TYPE sqrt_n = SQRT_FUN(float_n);

  /* Calculate mean using MKL */
  for (j = 0; j < m; j++) {
    mean[j] = SCALAR_VAL(0.0);
#ifdef DATA_TYPE_IS_DOUBLE
    mean[j] = cblas_dasum(n, &data[0][j], m) / float_n;
#elif defined(DATA_TYPE_IS_FLOAT)
    mean[j] = cblas_sasum(n, &data[0][j], m) / float_n;
#else
    /* Fallback for other data types */
    for (i = 0; i < n; i++)
      mean[j] += data[i][j];
    mean[j] /= float_n;
#endif
  }

  /* Calculate standard deviation using MKL */
  for (j = 0; j < m; j++) {
    stddev[j] = SCALAR_VAL(0.0);
    for (i = 0; i < n; i++)
      stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
    stddev[j] /= float_n;
    stddev[j] = SQRT_FUN(stddev[j]);
    stddev[j] = stddev[j] <= eps ? SCALAR_VAL(1.0) : stddev[j];
  }

  /* Center and reduce the column vectors */
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
      data[i][j] -= mean[j];
      data[i][j] /= sqrt_n * stddev[j];
    }

  /* Calculate the correlation matrix using MKL's GEMM */
  /* Set diagonal elements to 1.0 */
  for (i = 0; i < m; i++)
    corr[i][i] = SCALAR_VAL(1.0);

  /* Use GEMM to compute data^T * data which gives us correlation matrix */
#ifdef DATA_TYPE_IS_DOUBLE
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
              m, m, n,
              SCALAR_VAL(1.0), &data[0][0], m,
              &data[0][0], m,
              SCALAR_VAL(0.0), &corr[0][0], m);
#elif defined(DATA_TYPE_IS_FLOAT)
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
              m, m, n,
              SCALAR_VAL(1.0), &data[0][0], m,
              &data[0][0], m,
              SCALAR_VAL(0.0), &corr[0][0], m);
#else
  /* Fallback for other data types - use the naive implementation */
  for (i = 0; i < m-1; i++) {
    corr[i][i] = SCALAR_VAL(1.0);
    for (j = i+1; j < m; j++) {
      corr[i][j] = SCALAR_VAL(0.0);
      for (int k = 0; k < n; k++)
        corr[i][j] += (data[k][i] * data[k][j]);
      corr[j][i] = corr[i][j];
    }
  }
  corr[m-1][m-1] = SCALAR_VAL(1.0);
#endif
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int m,
                 DATA_TYPE POLYBENCH_2D(corr_naive,M,M,m,m),
                 DATA_TYPE POLYBENCH_2D(corr_mkl,M,M,m,m))
{
    int i, j;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            diff = fabs(corr_naive[i][j] - corr_mkl[i][j]);
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
double time_kernel(void (*kernel)(int, int, DATA_TYPE, 
                                 DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
                                 DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
                                 DATA_TYPE POLYBENCH_1D(mean,M,m),
                                 DATA_TYPE POLYBENCH_1D(stddev,M,m)),
                  int m, int n, DATA_TYPE float_n,
                  DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
                  DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
                  DATA_TYPE POLYBENCH_1D(mean,M,m),
                  DATA_TYPE POLYBENCH_1D(stddev,M,m))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(m, n, float_n, data, corr, mean, stddev);
    
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
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(data_mkl,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(corr,DATA_TYPE,M,M,m,m);
  POLYBENCH_2D_ARRAY_DECL(corr_mkl,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  POLYBENCH_1D_ARRAY_DECL(mean_mkl,DATA_TYPE,M,m);
  POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);
  POLYBENCH_1D_ARRAY_DECL(stddev_mkl,DATA_TYPE,M,m);

  /* Initialize array(s). */
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(data_mkl));

  /* Start timer for naive implementation. */
  double naive_time = time_kernel(kernel_correlation, m, n, float_n,
                                POLYBENCH_ARRAY(data),
                                POLYBENCH_ARRAY(corr),
                                POLYBENCH_ARRAY(mean),
                                POLYBENCH_ARRAY(stddev));
  
  printf("Naive Correlation Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_correlation_mkl, m, n, float_n,
                              POLYBENCH_ARRAY(data_mkl),
                              POLYBENCH_ARRAY(corr_mkl),
                              POLYBENCH_ARRAY(mean_mkl),
                              POLYBENCH_ARRAY(stddev_mkl));
  
  printf("MKL Correlation Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(m, POLYBENCH_ARRAY(corr), POLYBENCH_ARRAY(corr_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(corr_mkl)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(data_mkl);
  POLYBENCH_FREE_ARRAY(corr);
  POLYBENCH_FREE_ARRAY(corr_mkl);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(mean_mkl);
  POLYBENCH_FREE_ARRAY(stddev);
  POLYBENCH_FREE_ARRAY(stddev_mkl);

  return 0;
}
