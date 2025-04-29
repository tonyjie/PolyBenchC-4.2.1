/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* floyd-warshall.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "floyd-warshall.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(path,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(path_mkl,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      path[i][j] = i*j%7+1;
      if ((i+j)%13 == 0 || (i+j)%7==0 || (i+j)%11 == 0)
         path[i][j] = 999;
      path_mkl[i][j] = path[i][j];
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(path,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("path");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, path[i][j]);
    }
  POLYBENCH_DUMP_END("path");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_floyd_warshall(int n,
			   DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j, k;

#pragma scop
  for (k = 0; k < _PB_N; k++)
    {
      for(i = 0; i < _PB_N; i++)
	for (j = 0; j < _PB_N; j++)
	  path[i][j] = path[i][j] < path[i][k] + path[k][j] ?
	    path[i][j] : path[i][k] + path[k][j];
    }
#pragma endscop
}

/* MKL optimized implementation */
static
void kernel_floyd_warshall_mkl(int n,
			     DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j, k;
  
  /* Use MKL's vectorization where possible */
#ifdef DATA_TYPE_IS_FLOAT
  float* ik_path = (float*)mkl_malloc(n * sizeof(float), 64);
  float* kj_path = (float*)mkl_malloc(n * sizeof(float), 64);
  float* sum = (float*)mkl_malloc(n * sizeof(float), 64);
  
  for (k = 0; k < n; k++) {
    for (i = 0; i < n; i++) {
      /* Create a vector of path[i][k] values */
      for (j = 0; j < n; j++) {
        ik_path[j] = path[i][k];
      }
      
      /* Extract path[k][j] row */
      for (j = 0; j < n; j++) {
        kj_path[j] = path[k][j];
      }
      
      /* Compute ik_path + kj_path */
      #pragma omp simd
      for (j = 0; j < n; j++) {
        sum[j] = ik_path[j] + kj_path[j];
      }
      
      /* Update path[i][j] = min(path[i][j], sum[j]) */
      #pragma omp simd
      for (j = 0; j < n; j++) {
        path[i][j] = path[i][j] < sum[j] ? path[i][j] : sum[j];
      }
    }
  }
  
  mkl_free(ik_path);
  mkl_free(kj_path);
  mkl_free(sum);
  
#elif defined(DATA_TYPE_IS_DOUBLE)
  double* ik_path = (double*)mkl_malloc(n * sizeof(double), 64);
  double* kj_path = (double*)mkl_malloc(n * sizeof(double), 64);
  double* sum = (double*)mkl_malloc(n * sizeof(double), 64);
  
  for (k = 0; k < n; k++) {
    for (i = 0; i < n; i++) {
      /* Create a vector of path[i][k] values */
      for (j = 0; j < n; j++) {
        ik_path[j] = path[i][k];
      }
      
      /* Extract path[k][j] row */
      for (j = 0; j < n; j++) {
        kj_path[j] = path[k][j];
      }
      
      /* Compute ik_path + kj_path */
      #pragma omp simd
      for (j = 0; j < n; j++) {
        sum[j] = ik_path[j] + kj_path[j];
      }
      
      /* Update path[i][j] = min(path[i][j], sum[j]) */
      #pragma omp simd
      for (j = 0; j < n; j++) {
        path[i][j] = path[i][j] < sum[j] ? path[i][j] : sum[j];
      }
    }
  }
  
  mkl_free(ik_path);
  mkl_free(kj_path);
  mkl_free(sum);
  
#else
  /* For integer types, use a simpler MKL-enabled implementation */
  int* buffer = (int*)mkl_malloc(n * sizeof(int), 64);
  
  for (k = 0; k < n; k++) {
    for (i = 0; i < n; i++) {
      DATA_TYPE ik_val = path[i][k];
      
      /* Extract row for vectorized addition */
      for (j = 0; j < n; j++) {
        buffer[j] = ik_val + path[k][j];
      }
      
      /* Vectorized min operation */
      #pragma omp simd
      for (j = 0; j < n; j++) {
        path[i][j] = path[i][j] < buffer[j] ? path[i][j] : buffer[j];
      }
    }
  }
  
  mkl_free(buffer);
#endif
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int n,
                 DATA_TYPE POLYBENCH_2D(path_naive,N,N,n,n),
                 DATA_TYPE POLYBENCH_2D(path_mkl,N,N,n,n))
{
    int i, j;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            diff = fabs(path_naive[i][j] - path_mkl[i][j]);
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
double time_kernel(void (*kernel)(int, 
                                 DATA_TYPE POLYBENCH_2D(path,N,N,n,n)),
                  int n,
                  DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(n, path);
    
    /* End timer */
    end_time = get_time();
    
    return end_time - start_time;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(path, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(path_mkl, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(path), POLYBENCH_ARRAY(path_mkl));

  /* Start timer for naive implementation. */
  double naive_time = time_kernel(kernel_floyd_warshall, n, POLYBENCH_ARRAY(path));
  
  printf("Naive FLOYD-WARSHALL Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_floyd_warshall_mkl, n, POLYBENCH_ARRAY(path_mkl));
  
  printf("MKL FLOYD-WARSHALL Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(n, POLYBENCH_ARRAY(path), POLYBENCH_ARRAY(path_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(path)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(path);
  POLYBENCH_FREE_ARRAY(path_mkl);

  return 0;
}
