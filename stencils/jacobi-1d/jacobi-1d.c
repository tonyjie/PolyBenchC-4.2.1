/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* jacobi-1d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>
// #include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-1d.h"


/* Helper macros */
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n),
		 DATA_TYPE POLYBENCH_1D(B,N,n),
     DATA_TYPE POLYBENCH_1D(A_opt,N,n),
     DATA_TYPE POLYBENCH_1D(B_opt,N,n))
{
  int i;

  for (i = 0; i < n; i++)
      {
	A[i] = ((DATA_TYPE) i+ 2) / n;
	B[i] = ((DATA_TYPE) i+ 3) / n;
  A_opt[i] = A[i];
  B_opt[i] = B[i];
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    {
      if (i % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


// Optimized kernel with temporal and spatial tiling
static void kernel_jacobi_1d_optimized(int tsteps, int n,
                                    DATA_TYPE POLYBENCH_1D(A,N,n),
                                    DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int t, i, tt, ii;
  const int time_tile_size = 8;  // Time tile size
  const int space_tile_size = 256;  // Space tile size, tune for your L1 cache
  
  // Temporal and spatial tiling
  #pragma omp parallel for private(tt, t, ii, i)
  for (tt = 0; tt < tsteps; tt += time_tile_size) {
    for (ii = 1; ii < n-1; ii += space_tile_size) {
      int end_t = MIN(tt + time_tile_size, tsteps);
      int end_i = MIN(ii + space_tile_size, n-1);
      
      for (t = tt; t < end_t; t++) {
        // SIMD vectorization of inner loop
        #pragma omp simd
        for (i = MAX(1, ii); i < end_i; i++) {
          if (t % 2 == 0)
            B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1]);
          else
            A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1]);
        }
      }
    }
  }
}


void kernel_jacobi_1d_mkl(int tsteps, int n, 
                         DATA_TYPE POLYBENCH_1D(A,N,n),
                         DATA_TYPE POLYBENCH_1D(B,N,n)) {
    DATA_TYPE *temp = (DATA_TYPE *)malloc(n * sizeof(DATA_TYPE));

    for (int t = 0; t < tsteps; t++) {
        // Step 1: B = A[i-1] + A[i] + A[i+1] → shifted adds using daxpy
        // Initialize temp to 0
        cblas_dscal(n, 0.0, temp, 1);

        // Add A[i-1]
        cblas_daxpy(n - 2, 1.0, A, 1, temp + 1, 1);

        // Add A[i]
        cblas_daxpy(n - 2, 1.0, A + 1, 1, temp + 1, 1);

        // Add A[i+1]
        cblas_daxpy(n - 2, 1.0, A + 2, 1, temp + 1, 1);

        // Scale: B = 0.33333 * sum
        for (int i = 1; i < n - 1; i++) {
            B[i] = 0.33333 * temp[i];
        }

        // Repeat for A = f(B)
        cblas_dscal(n, 0.0, temp, 1);
        cblas_daxpy(n - 2, 1.0, B, 1, temp + 1, 1);
        cblas_daxpy(n - 2, 1.0, B + 1, 1, temp + 1, 1);
        cblas_daxpy(n - 2, 1.0, B + 2, 1, temp + 1, 1);
        for (int i = 1; i < n - 1; i++) {
            A[i] = 0.33333 * temp[i];
        }
    }

    free(temp);
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_jacobi_1d(int tsteps,
			    int n,
			    DATA_TYPE POLYBENCH_1D(A,N,n),
			    DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int t, i;

#pragma scop
  for (t = 0; t < _PB_TSTEPS; t++)
    {
      for (i = 1; i < _PB_N - 1; i++)
	B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
      for (i = 1; i < _PB_N - 1; i++)
	A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
    }
#pragma endscop

}


/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int n,
                 DATA_TYPE POLYBENCH_1D(A_ref,N,n),
                 DATA_TYPE POLYBENCH_1D(A_test,N,n))
{
    int i;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < n; i++) {
        diff = fabs(A_ref[i] - A_test[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    
    printf("Maximum difference: %e\n", max_diff);
    
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
double time_kernel(void (*kernel)(int, int, 
                                 DATA_TYPE POLYBENCH_1D(A,N,n),
                                 DATA_TYPE POLYBENCH_1D(B,N,n)),
                  int tsteps, int n,
                  DATA_TYPE POLYBENCH_1D(A,N,n),
                  DATA_TYPE POLYBENCH_1D(B,N,n))
{
    double start_time, end_time;
    double total_time = 0.0;
    int runs = 50;
    
    for (int run = 0; run < runs; run++) {
        /* Flush cache before timing */
        polybench_flush_cache();
        
        /* Start timer */
        start_time = get_time();
        
        /* Run kernel */
        kernel(tsteps, n, A, B);
        
        /* End timer */
        end_time = get_time();
        
        total_time += (end_time - start_time);
    }
    
    /* Return average time */
    return total_time / runs;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(A_opt, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B_opt, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(A_mkl, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B_mkl, DATA_TYPE, N, n);
  
  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(A_opt), POLYBENCH_ARRAY(B_opt));
  
  /* Copy arrays for MKL version */
  memcpy(POLYBENCH_ARRAY(A_mkl), POLYBENCH_ARRAY(A), n * sizeof(DATA_TYPE));
  memcpy(POLYBENCH_ARRAY(B_mkl), POLYBENCH_ARRAY(B), n * sizeof(DATA_TYPE));
  
  /* Start timer for naive implementation. */
  double naive_time = time_kernel(kernel_jacobi_1d, tsteps, n,
                                POLYBENCH_ARRAY(A),
                                POLYBENCH_ARRAY(B));
  
  printf("Naive implementation time: %0.6f seconds\n", naive_time);
  
  /* Start timer for optimized implementation. */
  double optimized_time = time_kernel(kernel_jacobi_1d_optimized, tsteps, n,
                                    POLYBENCH_ARRAY(A_opt),
                                    POLYBENCH_ARRAY(B_opt));
  
  printf("Optimized implementation time: %0.6f seconds\n", optimized_time);
  printf("Speedup (optimized vs naive): %.2fx\n", naive_time / optimized_time);
  
  /* Start timer for MKL implementation */
  double mkl_time = time_kernel(kernel_jacobi_1d_mkl, tsteps, n,
                              POLYBENCH_ARRAY(A_mkl),
                              POLYBENCH_ARRAY(B_mkl));
  
  printf("MKL implementation time: %0.6f seconds\n", mkl_time);
  printf("Speedup (MKL vs naive): %.2fx\n", naive_time / mkl_time);
  
  /* Verify the correctness of implementations */
  printf("\nVerifying results:\n");
  printf("Optimized vs Naive: ");
  verify_results(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_opt));
  printf("MKL vs Naive: ");
  verify_results(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(A_opt);
  POLYBENCH_FREE_ARRAY(B_opt);
  POLYBENCH_FREE_ARRAY(A_mkl);
  POLYBENCH_FREE_ARRAY(B_mkl);

  return 0;
}
