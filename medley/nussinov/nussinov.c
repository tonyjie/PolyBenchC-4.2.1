/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* nussinov.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "nussinov.h"

/* RNA bases represented as chars, range is [0,3] */
typedef char base;

#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0)
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

/* Array initialization. */
static
void init_array (int n,
                 base POLYBENCH_1D(seq,N,n),
		 DATA_TYPE POLYBENCH_2D(table,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(table_mkl,N,N,n,n))
{
  int i, j;

  //base is AGCT/0..3
  for (i=0; i <n; i++) {
     seq[i] = (base)((i+1)%4);
  }

  for (i=0; i <n; i++)
     for (j=0; j <n; j++) {
       table[i][j] = 0;
       table_mkl[i][j] = 0;
     }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(table,N,N,n,n))

{
  int i, j;
  int t = 0;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("table");
  for (i = 0; i < n; i++) {
    for (j = i; j < n; j++) {
      if (t % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, table[i][j]);
      t++;
    }
  }
  POLYBENCH_DUMP_END("table");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/*
  Original version by Dave Wonnacott at Haverford College <davew@cs.haverford.edu>,
  with help from Allison Lake, Ting Zhou, and Tian Jin,
  based on algorithm by Nussinov, described in Allison Lake's senior thesis.
*/
static
void kernel_nussinov(int n, base POLYBENCH_1D(seq,N,n),
			   DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
  int i, j, k;

#pragma scop
 for (i = _PB_N-1; i >= 0; i--) {
  for (j=i+1; j<_PB_N; j++) {

   if (j-1>=0)
      table[i][j] = max_score(table[i][j], table[i][j-1]);
   if (i+1<_PB_N)
      table[i][j] = max_score(table[i][j], table[i+1][j]);

   if (j-1>=0 && i+1<_PB_N) {
     /* don't allow adjacent elements to bond */
     if (i<j-1)
        table[i][j] = max_score(table[i][j], table[i+1][j-1]+match(seq[i], seq[j]));
     else
        table[i][j] = max_score(table[i][j], table[i+1][j-1]);
   }

   for (k=i+1; k<j; k++) {
      table[i][j] = max_score(table[i][j], table[i][k] + table[k+1][j]);
   }
  }
 }
#pragma endscop
}

/* MKL optimized implementation */
static
void kernel_nussinov_mkl(int n, base POLYBENCH_1D(seq,N,n),
                      DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
  int i, j, k;
  DATA_TYPE* max_vals = (DATA_TYPE*)mkl_malloc(n * sizeof(DATA_TYPE), 64);
  
  /* The basic structure remains the same, we'll optimize using vectorization where possible */
  for (i = n-1; i >= 0; i--) {
    for (j = i+1; j < n; j++) {
      
      /* First check for simple dependencies */
      if (j-1 >= 0)
        table[i][j] = max_score(table[i][j], table[i][j-1]);
      if (i+1 < n)
        table[i][j] = max_score(table[i][j], table[i+1][j]);
      
      /* Handle diagonal dependency with match score */
      if (j-1 >= 0 && i+1 < n) {
        if (i < j-1)
          table[i][j] = max_score(table[i][j], table[i+1][j-1] + match(seq[i], seq[j]));
        else
          table[i][j] = max_score(table[i][j], table[i+1][j-1]);
      }
      
      /* For the k-loop, we'll use vectorization where applicable */
      if (j - i > 16) { /* Only vectorize for longer ranges */
        int len = j - i - 1;
        
        /* Collect the values table[i][k] into a vector */
        for (k = 0; k < len; k++) {
          max_vals[k] = table[i][i+k+1] + table[i+k+1+1][j];
        }
        
        /* Find the maximum */
        DATA_TYPE max_val = table[i][j];
        #pragma simd reduction(max:max_val)
        for (k = 0; k < len; k++) {
          max_val = max_score(max_val, max_vals[k]);
        }
        
        table[i][j] = max_val;
      } else {
        /* For smaller ranges, use the original loop */
        for (k = i+1; k < j; k++) {
          table[i][j] = max_score(table[i][j], table[i][k] + table[k+1][j]);
        }
      }
    }
  }
  
  mkl_free(max_vals);
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int n,
                 DATA_TYPE POLYBENCH_2D(table_naive,N,N,n,n),
                 DATA_TYPE POLYBENCH_2D(table_mkl,N,N,n,n))
{
    int i, j;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            diff = fabs(table_naive[i][j] - table_mkl[i][j]);
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
                                 base POLYBENCH_1D(seq,N,n),
                                 DATA_TYPE POLYBENCH_2D(table,N,N,n,n)),
                  int n,
                  base POLYBENCH_1D(seq,N,n),
                  DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(n, seq, table);
    
    /* End timer */
    end_time = get_time();
    
    return end_time - start_time;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(seq, base, N, n);
  POLYBENCH_2D_ARRAY_DECL(table, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(table_mkl, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(seq), POLYBENCH_ARRAY(table), POLYBENCH_ARRAY(table_mkl));

  /* Start timer for naive implementation. */
  double naive_time = time_kernel(kernel_nussinov, n,
                                POLYBENCH_ARRAY(seq),
                                POLYBENCH_ARRAY(table));
  
  printf("Naive NUSSINOV Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_nussinov_mkl, n,
                              POLYBENCH_ARRAY(seq),
                              POLYBENCH_ARRAY(table_mkl));
  
  printf("MKL NUSSINOV Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(n, POLYBENCH_ARRAY(table), POLYBENCH_ARRAY(table_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(table)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(seq);
  POLYBENCH_FREE_ARRAY(table);
  POLYBENCH_FREE_ARRAY(table_mkl);

  return 0;
}
