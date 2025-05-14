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
    DATA_TYPE POLYBENCH_2D(C_mkl,NI,NJ,ni,nj),
    DATA_TYPE POLYBENCH_2D(C_tiled,NI,NJ,ni,nj),
    DATA_TYPE POLYBENCH_2D(C_tiled_orig,NI,NJ,ni,nj),
    DATA_TYPE POLYBENCH_2D(C_tiled_mkl,NI,NJ,ni,nj),
    DATA_TYPE POLYBENCH_2D(C_tiled_mkl_orig,NI,NJ,ni,nj))
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
    for (j = 0; j < nj; j++) {
      C_mkl[i][j] = C[i][j];
      C_tiled[i][j] = C[i][j];
      C_tiled_orig[i][j] = C[i][j];
      C_tiled_mkl[i][j] = C[i][j];
      C_tiled_mkl_orig[i][j] = C[i][j];
    }
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

/* Tiled GEMM kernel implementation */
static
void kernel_gemm_tiled(int ni, int nj, int nk,
                      DATA_TYPE alpha,
                      DATA_TYPE beta,
                      DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
                      DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                      DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                      int ti, int tj, int tk)
{
  int i, j, k, ii, jj, kk;

  /* Apply beta to C matrix */
  for (i = 0; i < _PB_NI; i++) {
    for (j = 0; j < _PB_NJ; j++) {
      C[i][j] *= beta;
    }
  }

  /* Tiled matrix multiplication with alpha */
  for (i = 0; i < _PB_NI; i += ti) {
    for (j = 0; j < _PB_NJ; j += tj) {
      for (k = 0; k < _PB_NK; k += tk) {
        /* Tile bounds with min to handle edge cases */
        int i_bound = (i + ti < _PB_NI) ? i + ti : _PB_NI;
        int j_bound = (j + tj < _PB_NJ) ? j + tj : _PB_NJ;
        int k_bound = (k + tk < _PB_NK) ? k + tk : _PB_NK;

        /* Compute on tiles */
        for (ii = i; ii < i_bound; ii++) {
          for (kk = k; kk < k_bound; kk++) {
            DATA_TYPE a_val = alpha * A[ii][kk];
            for (jj = j; jj < j_bound; jj++) {
              C[ii][jj] += a_val * B[kk][jj];
            }
          }
        }
      }
    }
  }
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
                 DATA_TYPE POLYBENCH_2D(C_reference,NI,NJ,ni,nj),
                 DATA_TYPE POLYBENCH_2D(C_test,NI,NJ,ni,nj))
{
    int i, j;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            diff = fabs(C_reference[i][j] - C_test[i][j]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    
    // printf("Maximum difference: %e\n", max_diff);
    
    if (max_diff < threshold) {
        // printf("Verification PASSED: Results match within threshold\n");
        return 1; /* Success */
    } else {
        // printf("Verification FAILED: Results differ beyond threshold\n");
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
    const int NUM_RUNS = 3;  /* Number of runs for more stable timing */
    double total_time = 0.0;
    double start_time, end_time;
    int run;
    
    for (run = 0; run < NUM_RUNS; run++) {
        /* Flush cache before timing */
        polybench_flush_cache();
        
        /* Start timer */
        start_time = get_time();
        
        /* Run kernel */
        kernel(ni, nj, nk, alpha, beta, C, A, B);
        
        /* End timer */
        end_time = get_time();
        
        total_time += (end_time - start_time);
    }
    
    return total_time / NUM_RUNS;  /* Return average time */
}

/* Function to time the tiled GEMM kernel */
static
double time_tiled_kernel(int ni, int nj, int nk,
                       DATA_TYPE alpha, DATA_TYPE beta,
                       DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
                       DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                       DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                       int ti, int tj, int tk)
{
    const int NUM_RUNS = 3;  /* Number of runs for more stable timing */
    double total_time = 0.0;
    double start_time, end_time;
    int run;
    
    for (run = 0; run < NUM_RUNS; run++) {
        /* Flush cache before timing */
        polybench_flush_cache();
        
        /* Start timer */
        start_time = get_time();
        
        /* Run tiled kernel */
        kernel_gemm_tiled(ni, nj, nk, alpha, beta, C, A, B, ti, tj, tk);
        
        /* End timer */
        end_time = get_time();
        
        total_time += (end_time - start_time);
    }
    
    return total_time / NUM_RUNS;  /* Return average time */
}

/* Function to perform exhaustive search over tile sizes */
static
void tile_size_search(int ni, int nj, int nk,
                     DATA_TYPE alpha, DATA_TYPE beta,
                     DATA_TYPE POLYBENCH_2D(C_ref,NI,NJ,ni,nj),
                     DATA_TYPE POLYBENCH_2D(C_tiled,NI,NJ,ni,nj),
                     DATA_TYPE POLYBENCH_2D(C_tiled_orig,NI,NJ,ni,nj),
                     DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                     DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
 
    /* Search space for tile sizes */
    int tile_sizes[] = {4, 8, 16, 32, 64, 128, 256};
    int num_tile_sizes = sizeof(tile_sizes) / sizeof(tile_sizes[0]);
    
    /* Best configuration found so far */
    int best_ti = 0, best_tj = 0, best_tk = 0;
    double best_time = 1e10;  /* Initialize to a large value */
    
    printf("\n=== Exhaustive Search for Optimal Tile Sizes ===\n");
    printf("Tile size combinations to explore: %d\n", num_tile_sizes * num_tile_sizes * num_tile_sizes);
    printf("%-6s %-6s %-6s %-12s %-10s\n", "Ti", "Tj", "Tk", "Time (sec)", "Verified");
    
    /* Exhaustive search over all tile size combinations */
    int ti_idx, tj_idx, tk_idx;
    for (ti_idx = 0; ti_idx < num_tile_sizes; ti_idx++) {
        int ti = tile_sizes[ti_idx];
        for (tj_idx = 0; tj_idx < num_tile_sizes; tj_idx++) {
            int tj = tile_sizes[tj_idx];
            for (tk_idx = 0; tk_idx < num_tile_sizes; tk_idx++) {
                int tk = tile_sizes[tk_idx];
                
                /* Reset C_tiled to original values */
                int i, j;
                for (i = 0; i < ni; i++) {
                    for (j = 0; j < nj; j++) {
                        C_tiled[i][j] = C_tiled_orig[i][j];
                    }
                }
                
                /* Time the tiled kernel with current tile sizes */
                double time = time_tiled_kernel(ni, nj, nk, alpha, beta, C_tiled, A, B, ti, tj, tk);
                
                /* Verify the result */
                int verified = verify_results(ni, nj, C_ref, C_tiled);
                
                /* Print current configuration and timing */
                printf("%-6d %-6d %-6d %-12.6f %-10s\n", ti, tj, tk, time, verified ? "Pass" : "Fail");
                
                /* Update best configuration if this one is better */
                if (verified && time < best_time) {
                    best_time = time;
                    best_ti = ti;
                    best_tj = tj;
                    best_tk = tk;
                }
            }
        }
    }
    
    /* Report best tile sizes found */
    printf("\n=== Best Tile Configuration Found ===\n");
    printf("Best tile sizes: Ti=%d, Tj=%d, Tk=%d\n", best_ti, best_tj, best_tk);
    printf("Best time: %0.6f seconds\n", best_time);
    
    /* Use the best configuration for final timing */
    // reset C_tiled to original values
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            C_tiled[i][j] = C_tiled_orig[i][j];
        }
    }
    
    /* Time the best configuration one more time */
    double final_time = time_tiled_kernel(ni, nj, nk, alpha, beta, C_tiled, A, B, best_ti, best_tj, best_tk);
    printf("Final verification of best configuration: %s\n", 
          verify_results(ni, nj, C_ref, C_tiled) ? "Passed" : "Failed");
    printf("Confirmed time with best configuration: %0.6f seconds\n", final_time);
}

/* Tiled GEMM kernel implementation with MKL calls for each tile */
static
void kernel_gemm_tiled_mkl(int ni, int nj, int nk,
                          DATA_TYPE alpha,
                          DATA_TYPE beta,
                          DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
                          DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                          DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                          int ti, int tj, int tk)
{
  int i, j, k;

  /* Apply beta to C matrix first */
  for (i = 0; i < _PB_NI; i++) {
    for (j = 0; j < _PB_NJ; j++) {
      C[i][j] *= beta;
    }
  }

  /* Tiled matrix multiplication with alpha using MKL for each tile */
  for (i = 0; i < _PB_NI; i += ti) {
    for (j = 0; j < _PB_NJ; j += tj) {
      for (k = 0; k < _PB_NK; k += tk) {
        /* Tile bounds with min to handle edge cases */
        int i_bound = (i + ti < _PB_NI) ? i + ti : _PB_NI;
        int j_bound = (j + tj < _PB_NJ) ? j + tj : _PB_NJ;
        int k_bound = (k + tk < _PB_NK) ? k + tk : _PB_NK;
        
        /* Actual tile dimensions */
        int ti_actual = i_bound - i;
        int tj_actual = j_bound - j;
        int tk_actual = k_bound - k;
        
        /* Use MKL for this tile: C(i:i_bound, j:j_bound) += A(i:i_bound, k:k_bound) * B(k:k_bound, j:j_bound) */
#ifdef DATA_TYPE_IS_DOUBLE
        /* Use MKL's double-precision GEMM: dgemm */
        cblas_dgemm(CblasRowMajor,     /* Matrix storage order: Row-major (C-style) */
                    CblasNoTrans,       /* A: no transpose */
                    CblasNoTrans,       /* B: no transpose */
                    ti_actual, tj_actual, tk_actual,  /* Tile dimensions */
                    alpha,              /* Alpha scalar */
                    &A[i][k], nk,       /* Tile of matrix A and leading dimension */
                    &B[k][j], nj,       /* Tile of matrix B and leading dimension */
                    1.0,                /* Beta=1.0 since we've already applied beta earlier */
                    &C[i][j], nj);      /* Tile of matrix C and leading dimension */
#elif defined(DATA_TYPE_IS_FLOAT)
        /* Use MKL's single-precision GEMM: sgemm */
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    ti_actual, tj_actual, tk_actual,
                    alpha,
                    &A[i][k], nk,
                    &B[k][j], nj,
                    1.0,
                    &C[i][j], nj);
#else
        /* Fallback for non-floating-point types */
        for (int ii = i; ii < i_bound; ii++) {
          for (int kk = k; kk < k_bound; kk++) {
            DATA_TYPE a_val = alpha * A[ii][kk];
            for (int jj = j; jj < j_bound; jj++) {
              C[ii][jj] += a_val * B[kk][jj];
            }
          }
        }
#endif
      }
    }
  }
}

/* Function to time the tiled MKL kernel */
static
double time_tiled_mkl_kernel(int ni, int nj, int nk,
                           DATA_TYPE alpha, DATA_TYPE beta,
                           DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
                           DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                           DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                           int ti, int tj, int tk)
{
    const int NUM_RUNS = 3;  /* Number of runs for more stable timing */
    double total_time = 0.0;
    double start_time, end_time;
    int run;
    
    for (run = 0; run < NUM_RUNS; run++) {
        /* Flush cache before timing */
        polybench_flush_cache();
        
        /* Start timer */
        start_time = get_time();
        
        /* Run tiled MKL kernel */
        kernel_gemm_tiled_mkl(ni, nj, nk, alpha, beta, C, A, B, ti, tj, tk);
        
        /* End timer */
        end_time = get_time();
        
        total_time += (end_time - start_time);
    }
    
    return total_time / NUM_RUNS;  /* Return average time */
}

/* Function to perform exhaustive search for tiled MKL */
static
void tiled_mkl_search(int ni, int nj, int nk,
                     DATA_TYPE alpha, DATA_TYPE beta,
                     DATA_TYPE POLYBENCH_2D(C_ref,NI,NJ,ni,nj),
                     DATA_TYPE POLYBENCH_2D(C_tiled_mkl,NI,NJ,ni,nj),
                     DATA_TYPE POLYBENCH_2D(C_tiled_mkl_orig,NI,NJ,ni,nj),
                     DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
                     DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
                     double mkl_time)
{
    /* Search space for tile sizes - using powers of 2 */
    int tile_sizes[] = {16, 32, 64, 128, 256, 512, 1024};
    int num_tile_sizes = sizeof(tile_sizes) / sizeof(tile_sizes[0]);
    
    /* Best configuration found so far */
    int best_ti = 0, best_tj = 0, best_tk = 0;
    double best_time = 1e10;  /* Initialize to a large value */
    
    printf("\n=== Exhaustive Search for Optimal Tiled MKL Configuration ===\n");
    printf("Tile size combinations to explore: %d\n", num_tile_sizes * num_tile_sizes * num_tile_sizes);
    printf("%-6s %-6s %-6s %-12s %-10s %-15s\n", "Ti", "Tj", "Tk", "Time (sec)", "Verified", "vs Direct MKL");
    
    /* Exhaustive search over all tile size combinations */
    int ti_idx, tj_idx, tk_idx;
    for (ti_idx = 0; ti_idx < num_tile_sizes; ti_idx++) {
        int ti = tile_sizes[ti_idx];
        /* Skip if tile size is larger than matrix dimension */
        if (ti > ni) continue;
        
        for (tj_idx = 0; tj_idx < num_tile_sizes; tj_idx++) {
            int tj = tile_sizes[tj_idx];
            if (tj > nj) continue;
            
            for (tk_idx = 0; tk_idx < num_tile_sizes; tk_idx++) {
                int tk = tile_sizes[tk_idx];
                if (tk > nk) continue;
                
                /* Reset C_tiled_mkl to original values */
                int i, j;
                for (i = 0; i < ni; i++) {
                    for (j = 0; j < nj; j++) {
                        C_tiled_mkl[i][j] = C_tiled_mkl_orig[i][j];
                    }
                }
                
                /* Time the tiled MKL kernel with current tile sizes */
                double time = time_tiled_mkl_kernel(ni, nj, nk, alpha, beta, 
                                                  C_tiled_mkl, A, B, ti, tj, tk);
                
                /* Verify the result */
                int verified = verify_results(ni, nj, C_ref, C_tiled_mkl);
                
                /* Calculate speedup/slowdown compared to direct MKL */
                double vs_mkl = mkl_time / time;
                
                /* Print current configuration and timing */
                printf("%-6d %-6d %-6d %-12.6f %-10s %-15.3f\n", 
                      ti, tj, tk, time, verified ? "Pass" : "Fail", vs_mkl);
                
                /* Update best configuration if this one is better */
                if (verified && time < best_time) {
                    best_time = time;
                    best_ti = ti;
                    best_tj = tj;
                    best_tk = tk;
                }
            }
        }
    }
    
    /* Report best tile sizes found */
    printf("\n=== Best Tiled MKL Configuration Found ===\n");
    if (best_time < 1e10) {
        printf("Best tile sizes: Ti=%d, Tj=%d, Tk=%d\n", best_ti, best_tj, best_tk);
        printf("Best time: %0.6f seconds\n", best_time);
        printf("Performance vs direct MKL: %.3f (%s)\n", 
              mkl_time / best_time, 
              (mkl_time > best_time) ? "FASTER than direct MKL" : "SLOWER than direct MKL");
        
        /* Use the best configuration for final timing */
        for (int i = 0; i < ni; i++) {
            for (int j = 0; j < nj; j++) {
                C_tiled_mkl[i][j] = C_tiled_mkl_orig[i][j];
            }
        }
        
        /* Time the best configuration one more time */
        double final_time = time_tiled_mkl_kernel(ni, nj, nk, alpha, beta, 
                                                C_tiled_mkl, A, B, best_ti, best_tj, best_tk);
        printf("Final verification of best configuration: %s\n", 
              verify_results(ni, nj, C_ref, C_tiled_mkl) ? "Passed" : "Failed");
        printf("Confirmed time with best configuration: %0.6f seconds\n", final_time);
        printf("Confirmed performance vs direct MKL: %.3f (%s)\n", 
              mkl_time / final_time, 
              (mkl_time > final_time) ? "FASTER than direct MKL" : "SLOWER than direct MKL");
    } else {
        printf("No valid configurations found!\n");
    }
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
  
  /* For tiled implementation */
  POLYBENCH_2D_ARRAY_DECL(C_tiled,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(C_tiled_orig,DATA_TYPE,NI,NJ,ni,nj); // used for resetting C_tiled to original values
  
  /* For tiled MKL implementation */
  POLYBENCH_2D_ARRAY_DECL(C_tiled_mkl,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(C_tiled_mkl_orig,DATA_TYPE,NI,NJ,ni,nj);

  /* Initialize array(s). */
  init_array (ni, nj, nk, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C_mkl),
	      POLYBENCH_ARRAY(C_tiled),
	      POLYBENCH_ARRAY(C_tiled_orig),
	      POLYBENCH_ARRAY(C_tiled_mkl),
	      POLYBENCH_ARRAY(C_tiled_mkl_orig));


  /* Start timer for naive implementation. */
  double naive_time = time_kernel(kernel_gemm, ni, nj, nk, alpha, beta,
                                POLYBENCH_ARRAY(C),
                                POLYBENCH_ARRAY(A),
                                POLYBENCH_ARRAY(B));
  
  printf("\n=== Performance Results ===\n");
  printf("Naive GEMM Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_gemm_mkl, ni, nj, nk, alpha, beta,
                              POLYBENCH_ARRAY(C_mkl),
                              POLYBENCH_ARRAY(A),
                              POLYBENCH_ARRAY(B));
  
  printf("MKL GEMM Time: %0.6f seconds\n", mkl_time);
  printf("MKL Speedup vs Naive: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  printf("\n=== MKL Verification ===\n");
  int mkl_verification = verify_results(ni, nj, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_mkl));
  printf("MKL Verification: %s\n", mkl_verification ? "Passed" : "Failed");
  
  // /* Run exhaustive search for optimal tile sizes */
  // printf("\n=== Starting Tile Size Search ===\n");
  // tile_size_search(ni, nj, nk, alpha, beta, 
  //                 POLYBENCH_ARRAY(C), 
  //                 POLYBENCH_ARRAY(C_tiled),
  //                 POLYBENCH_ARRAY(C_tiled_orig),
  //                 POLYBENCH_ARRAY(A),
  //                 POLYBENCH_ARRAY(B));
  
  /* Run exhaustive search for optimal tiled MKL configuration */
  printf("\n=== Starting Tiled MKL Search ===\n");
  tiled_mkl_search(ni, nj, nk, alpha, beta,
                  POLYBENCH_ARRAY(C),
                  POLYBENCH_ARRAY(C_tiled_mkl),
                  POLYBENCH_ARRAY(C_tiled_mkl_orig),
                  POLYBENCH_ARRAY(A),
                  POLYBENCH_ARRAY(B),
                  mkl_time);

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(C_mkl);
  POLYBENCH_FREE_ARRAY(C_tiled);
  POLYBENCH_FREE_ARRAY(C_tiled_orig);
  POLYBENCH_FREE_ARRAY(C_tiled_mkl);
  POLYBENCH_FREE_ARRAY(C_tiled_mkl_orig);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
