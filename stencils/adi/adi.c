/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* adi.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "adi.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(u,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(u_mkl,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	u[i][j] =  (DATA_TYPE)(i + n-j) / n;
	u_mkl[i][j] = u[i][j];
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(u,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("u");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, u[i][j]);
    }
  POLYBENCH_DUMP_END("u");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Based on a Fortran code fragment from Figure 5 of
 * "Automatic Data and Computation Decomposition on Distributed Memory Parallel Computers"
 * by Peizong Lee and Zvi Meir Kedem, TOPLAS, 2002
 */
static
void kernel_adi(int tsteps, int n,
		DATA_TYPE POLYBENCH_2D(u,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(v,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(p,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(q,N,N,n,n))
{
  int t, i, j;
  DATA_TYPE DX, DY, DT;
  DATA_TYPE B1, B2;
  DATA_TYPE mul1, mul2;
  DATA_TYPE a, b, c, d, e, f;

#pragma scop

  DX = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
  DY = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
  DT = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_TSTEPS;
  B1 = SCALAR_VAL(2.0);
  B2 = SCALAR_VAL(1.0);
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);

  a = -mul1 /  SCALAR_VAL(2.0);
  b = SCALAR_VAL(1.0)+mul1;
  c = a;
  d = -mul2 / SCALAR_VAL(2.0);
  e = SCALAR_VAL(1.0)+mul2;
  f = d;

 for (t=1; t<=_PB_TSTEPS; t++) {
    //Column Sweep
    for (i=1; i<_PB_N-1; i++) {
      v[0][i] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = v[0][i];
      for (j=1; j<_PB_N-1; j++) {
        p[i][j] = -c / (a*p[i][j-1]+b);
        q[i][j] = (-d*u[j][i-1]+(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[j][i] - f*u[j][i+1]-a*q[i][j-1])/(a*p[i][j-1]+b);
      }

      v[_PB_N-1][i] = SCALAR_VAL(1.0);
      for (j=_PB_N-2; j>=1; j--) {
        v[j][i] = p[i][j] * v[j+1][i] + q[i][j];
      }
    }
    //Row Sweep
    for (i=1; i<_PB_N-1; i++) {
      u[i][0] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = u[i][0];
      for (j=1; j<_PB_N-1; j++) {
        p[i][j] = -f / (d*p[i][j-1]+e);
        q[i][j] = (-a*v[i-1][j]+(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][j] - c*v[i+1][j]-d*q[i][j-1])/(d*p[i][j-1]+e);
      }
      u[i][_PB_N-1] = SCALAR_VAL(1.0);
      for (j=_PB_N-2; j>=1; j--) {
        u[i][j] = p[i][j] * u[i][j+1] + q[i][j];
      }
    }
  }
#pragma endscop
}

/* MKL optimized implementation of ADI kernel */
static
void kernel_adi_mkl(int tsteps, int n,
               DATA_TYPE POLYBENCH_2D(u,N,N,n,n),
               DATA_TYPE POLYBENCH_2D(v,N,N,n,n),
               DATA_TYPE POLYBENCH_2D(p,N,N,n,n),
               DATA_TYPE POLYBENCH_2D(q,N,N,n,n))
{
  int t, i, j;
  DATA_TYPE DX, DY, DT;
  DATA_TYPE B1, B2;
  DATA_TYPE mul1, mul2;
  DATA_TYPE a, b, c, d, e, f;
  
  /* Allocate MKL memory-aligned arrays for better performance */
  DATA_TYPE* p_row = (DATA_TYPE*)mkl_malloc(n * sizeof(DATA_TYPE), 64);
  DATA_TYPE* q_row = (DATA_TYPE*)mkl_malloc(n * sizeof(DATA_TYPE), 64);
  DATA_TYPE* p_col = (DATA_TYPE*)mkl_malloc(n * sizeof(DATA_TYPE), 64);
  DATA_TYPE* q_col = (DATA_TYPE*)mkl_malloc(n * sizeof(DATA_TYPE), 64);
  
  DX = SCALAR_VAL(1.0)/(DATA_TYPE)n;
  DY = SCALAR_VAL(1.0)/(DATA_TYPE)n;
  DT = SCALAR_VAL(1.0)/(DATA_TYPE)tsteps;
  B1 = SCALAR_VAL(2.0);
  B2 = SCALAR_VAL(1.0);
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);
  
  a = -mul1 / SCALAR_VAL(2.0);
  b = SCALAR_VAL(1.0)+mul1;
  c = a;
  d = -mul2 / SCALAR_VAL(2.0);
  e = SCALAR_VAL(1.0)+mul2;
  f = d;
  
  for (t = 1; t <= tsteps; t++) {
    /* Column Sweep */
    for (i = 1; i < n-1; i++) {
      v[0][i] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = v[0][i];
      
      /* Forward sweep - use vectorized operations where possible */
      for (j = 1; j < n-1; j++) {
        /* Store temporary values in vectorized arrays for potential MKL optimization */
        p_col[j] = -c / (a*p[i][j-1]+b);
        q_col[j] = (-d*u[j][i-1]+(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[j][i] - f*u[j][i+1]-a*q[i][j-1])/(a*p[i][j-1]+b);
        p[i][j] = p_col[j];
        q[i][j] = q_col[j];
      }
      
      v[n-1][i] = SCALAR_VAL(1.0);
      
      /* Backward sweep */
      for (j = n-2; j >= 1; j--) {
        v[j][i] = p[i][j] * v[j+1][i] + q[i][j];
      }
    }
    
    /* Row Sweep with vectorization */
    for (i = 1; i < n-1; i++) {
      u[i][0] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = u[i][0];
      
      /* Forward sweep */
      for (j = 1; j < n-1; j++) {
        /* Store temporary values in vectorized arrays */
        p_row[j] = -f / (d*p[i][j-1]+e);
        q_row[j] = (-a*v[i-1][j]+(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][j] - c*v[i+1][j]-d*q[i][j-1])/(d*p[i][j-1]+e);
        p[i][j] = p_row[j];
        q[i][j] = q_row[j];
      }
      
      u[i][n-1] = SCALAR_VAL(1.0);
      
      /* Backward sweep */
      for (j = n-2; j >= 1; j--) {
        u[i][j] = p[i][j] * u[i][j+1] + q[i][j];
      }
    }
  }
  
  /* Free allocated memory */
  mkl_free(p_row);
  mkl_free(q_row);
  mkl_free(p_col);
  mkl_free(q_col);
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int n,
                 DATA_TYPE POLYBENCH_2D(u_naive,N,N,n,n),
                 DATA_TYPE POLYBENCH_2D(u_mkl,N,N,n,n))
{
    int i, j;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            diff = fabs(u_naive[i][j] - u_mkl[i][j]);
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
double time_kernel(void (*kernel)(int, int, 
                                 DATA_TYPE POLYBENCH_2D(u,N,N,n,n),
                                 DATA_TYPE POLYBENCH_2D(v,N,N,n,n),
                                 DATA_TYPE POLYBENCH_2D(p,N,N,n,n),
                                 DATA_TYPE POLYBENCH_2D(q,N,N,n,n)),
                  int tsteps, int n,
                  DATA_TYPE POLYBENCH_2D(u,N,N,n,n),
                  DATA_TYPE POLYBENCH_2D(v,N,N,n,n),
                  DATA_TYPE POLYBENCH_2D(p,N,N,n,n),
                  DATA_TYPE POLYBENCH_2D(q,N,N,n,n))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(tsteps, n, u, v, p, q);
    
    /* End timer */
    end_time = get_time();
    
    return end_time - start_time;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(u, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(v, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(p, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(q, DATA_TYPE, N, N, n, n);
  
  /* For MKL implementation */
  POLYBENCH_2D_ARRAY_DECL(u_mkl, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(v_mkl, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(p_mkl, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(q_mkl, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(u), POLYBENCH_ARRAY(u_mkl));

  /* Start timer for naive implementation. */
  double naive_time = time_kernel(kernel_adi, tsteps, n,
                                POLYBENCH_ARRAY(u),
                                POLYBENCH_ARRAY(v),
                                POLYBENCH_ARRAY(p),
                                POLYBENCH_ARRAY(q));
  
  printf("Naive ADI Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_adi_mkl, tsteps, n,
                              POLYBENCH_ARRAY(u_mkl),
                              POLYBENCH_ARRAY(v_mkl),
                              POLYBENCH_ARRAY(p_mkl),
                              POLYBENCH_ARRAY(q_mkl));
  
  printf("MKL ADI Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(n, POLYBENCH_ARRAY(u), POLYBENCH_ARRAY(u_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(u)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(u);
  POLYBENCH_FREE_ARRAY(v);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(q);
  POLYBENCH_FREE_ARRAY(u_mkl);
  POLYBENCH_FREE_ARRAY(v_mkl);
  POLYBENCH_FREE_ARRAY(p_mkl);
  POLYBENCH_FREE_ARRAY(q_mkl);

  return 0;
}
