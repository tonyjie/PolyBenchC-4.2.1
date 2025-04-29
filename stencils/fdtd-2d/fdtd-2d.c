/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* fdtd-2d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "fdtd-2d.h"


/* Array initialization. */
static
void init_array (int tmax,
		 int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax),
		 DATA_TYPE POLYBENCH_2D(ex_mkl,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey_mkl,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz_mkl,NX,NY,nx,ny))
{
  int i, j;

  for (i = 0; i < tmax; i++)
    _fict_[i] = (DATA_TYPE) i;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      {
	ex[i][j] = ((DATA_TYPE) i*(j+1)) / nx;
	ey[i][j] = ((DATA_TYPE) i*(j+2)) / ny;
	hz[i][j] = ((DATA_TYPE) i*(j+3)) / nx;
	ex_mkl[i][j] = ex[i][j];
	ey_mkl[i][j] = ey[i][j];
	hz_mkl[i][j] = hz[i][j];
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("ex");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ex[i][j]);
    }
  POLYBENCH_DUMP_END("ex");
  POLYBENCH_DUMP_FINISH;

  POLYBENCH_DUMP_BEGIN("ey");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ey[i][j]);
    }
  POLYBENCH_DUMP_END("ey");

  POLYBENCH_DUMP_BEGIN("hz");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, hz[i][j]);
    }
  POLYBENCH_DUMP_END("hz");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_fdtd_2d(int tmax,
		    int nx,
		    int ny,
		    DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int t, i, j;

#pragma scop

  for(t = 0; t < _PB_TMAX; t++)
    {
      for (j = 0; j < _PB_NY; j++)
	ey[0][j] = _fict_[t];
      for (i = 1; i < _PB_NX; i++)
	for (j = 0; j < _PB_NY; j++)
	  ey[i][j] = ey[i][j] - SCALAR_VAL(0.5)*(hz[i][j]-hz[i-1][j]);
      for (i = 0; i < _PB_NX; i++)
	for (j = 1; j < _PB_NY; j++)
	  ex[i][j] = ex[i][j] - SCALAR_VAL(0.5)*(hz[i][j]-hz[i][j-1]);
      for (i = 0; i < _PB_NX - 1; i++)
	for (j = 0; j < _PB_NY - 1; j++)
	  hz[i][j] = hz[i][j] - SCALAR_VAL(0.7)*  (ex[i][j+1] - ex[i][j] +
				       ey[i+1][j] - ey[i][j]);
    }

#pragma endscop
}

/* MKL optimized implementation of FDTD-2D kernel */
static
void kernel_fdtd_2d_mkl(int tmax,
                    int nx,
                    int ny,
                    DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
                    DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
                    DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
                    DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int t, i, j;
  
  /* Allocate MKL memory-aligned temporary arrays */
  DATA_TYPE* temp_ex = (DATA_TYPE*)mkl_malloc(nx * sizeof(DATA_TYPE), 64);
  DATA_TYPE* temp_ey = (DATA_TYPE*)mkl_malloc(ny * sizeof(DATA_TYPE), 64);
  DATA_TYPE* temp_hz = (DATA_TYPE*)mkl_malloc(nx * sizeof(DATA_TYPE), 64);
  
  for(t = 0; t < tmax; t++)
  {
    /* Set boundary condition */
    for (j = 0; j < ny; j++)
      ey[0][j] = _fict_[t];
    
    /* Update ey using vector operations where possible */
    for (i = 1; i < nx; i++) {
      /* Prepare vector for vector subtraction */
      #ifdef DATA_TYPE_IS_DOUBLE
      vdSub(ny, &hz[i][0], &hz[i-1][0], temp_hz); /* temp_hz = hz[i] - hz[i-1] */
      cblas_dscal(ny, -0.5, temp_hz, 1);         /* temp_hz = -0.5 * temp_hz */
      vdAdd(ny, &ey[i][0], temp_hz, &ey[i][0]); /* ey[i] = ey[i] + temp_hz */
      #elif defined(DATA_TYPE_IS_FLOAT)
      vsSub(ny, &hz[i][0], &hz[i-1][0], temp_hz);
      cblas_sscal(ny, -0.5f, temp_hz, 1);
      vsAdd(ny, &ey[i][0], temp_hz, &ey[i][0]);
      #else
      /* Fallback to scalar code for integer types */
      for (j = 0; j < ny; j++) {
        ey[i][j] = ey[i][j] - SCALAR_VAL(0.5) * (hz[i][j] - hz[i-1][j]);
      }
      #endif
    }
    
    /* Update ex */
    for (i = 0; i < nx; i++) {
      /* First element is special case due to boundary */
      ex[i][0] = ex[i][0];
      
      /* Vector operation for the rest of the row */
      #ifdef DATA_TYPE_IS_DOUBLE
      for (j = 1; j < ny; j++) {
        ex[i][j] = ex[i][j] - SCALAR_VAL(0.5) * (hz[i][j] - hz[i][j-1]);
      }
      #elif defined(DATA_TYPE_IS_FLOAT)
      for (j = 1; j < ny; j++) {
        ex[i][j] = ex[i][j] - SCALAR_VAL(0.5) * (hz[i][j] - hz[i][j-1]);
      }
      #else
      for (j = 1; j < ny; j++) {
        ex[i][j] = ex[i][j] - SCALAR_VAL(0.5) * (hz[i][j] - hz[i][j-1]);
      }
      #endif
    }
    
    /* Update hz - this is more challenging to vectorize fully */
    for (i = 0; i < nx - 1; i++) {
      for (j = 0; j < ny - 1; j++) {
        hz[i][j] = hz[i][j] - SCALAR_VAL(0.7) * (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j]);
      }
    }
  }
  
  /* Free allocated memory */
  mkl_free(temp_ex);
  mkl_free(temp_ey);
  mkl_free(temp_hz);
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int nx, int ny,
                 DATA_TYPE POLYBENCH_2D(ex_naive,NX,NY,nx,ny),
                 DATA_TYPE POLYBENCH_2D(ey_naive,NX,NY,nx,ny),
                 DATA_TYPE POLYBENCH_2D(hz_naive,NX,NY,nx,ny),
                 DATA_TYPE POLYBENCH_2D(ex_mkl,NX,NY,nx,ny),
                 DATA_TYPE POLYBENCH_2D(ey_mkl,NX,NY,nx,ny),
                 DATA_TYPE POLYBENCH_2D(hz_mkl,NX,NY,nx,ny))
{
    int i, j;
    DATA_TYPE diff_ex, diff_ey, diff_hz;
    DATA_TYPE max_diff_ex = 0.0;
    DATA_TYPE max_diff_ey = 0.0;
    DATA_TYPE max_diff_hz = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            diff_ex = fabs(ex_naive[i][j] - ex_mkl[i][j]);
            diff_ey = fabs(ey_naive[i][j] - ey_mkl[i][j]);
            diff_hz = fabs(hz_naive[i][j] - hz_mkl[i][j]);
            
            if (diff_ex > max_diff_ex) max_diff_ex = diff_ex;
            if (diff_ey > max_diff_ey) max_diff_ey = diff_ey;
            if (diff_hz > max_diff_hz) max_diff_hz = diff_hz;
        }
    }
    
    printf("Maximum difference between naive and MKL implementation:\n");
    printf("ex: %e\n", max_diff_ex);
    printf("ey: %e\n", max_diff_ey);
    printf("hz: %e\n", max_diff_hz);
    
    if (max_diff_ex < threshold && max_diff_ey < threshold && max_diff_hz < threshold) {
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
                                 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
                                 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
                                 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
                                 DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax)),
                  int tmax, int nx, int ny,
                  DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
                  DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
                  DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
                  DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(tmax, nx, ny, ex, ey, hz, _fict_);
    
    /* End timer */
    end_time = get_time();
    
    return end_time - start_time;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int tmax = TMAX;
  int nx = NX;
  int ny = NY;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(ex,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(ey,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(hz,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_1D_ARRAY_DECL(_fict_,DATA_TYPE,TMAX,tmax);
  
  /* For MKL implementation */
  POLYBENCH_2D_ARRAY_DECL(ex_mkl,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(ey_mkl,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(hz_mkl,DATA_TYPE,NX,NY,nx,ny);

  /* Initialize array(s). */
  init_array (tmax, nx, ny,
	      POLYBENCH_ARRAY(ex),
	      POLYBENCH_ARRAY(ey),
	      POLYBENCH_ARRAY(hz),
	      POLYBENCH_ARRAY(_fict_),
          POLYBENCH_ARRAY(ex_mkl),
          POLYBENCH_ARRAY(ey_mkl),
          POLYBENCH_ARRAY(hz_mkl));

  /* Start timer for naive implementation. */
  double naive_time = time_kernel(kernel_fdtd_2d, tmax, nx, ny,
                                POLYBENCH_ARRAY(ex),
                                POLYBENCH_ARRAY(ey),
                                POLYBENCH_ARRAY(hz),
                                POLYBENCH_ARRAY(_fict_));
  
  printf("Naive FDTD-2D Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_fdtd_2d_mkl, tmax, nx, ny,
                              POLYBENCH_ARRAY(ex_mkl),
                              POLYBENCH_ARRAY(ey_mkl),
                              POLYBENCH_ARRAY(hz_mkl),
                              POLYBENCH_ARRAY(_fict_));
  
  printf("MKL FDTD-2D Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(nx, ny, 
               POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz),
               POLYBENCH_ARRAY(ex_mkl), POLYBENCH_ARRAY(ey_mkl), POLYBENCH_ARRAY(hz_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nx, ny, POLYBENCH_ARRAY(ex),
				    POLYBENCH_ARRAY(ey),
				    POLYBENCH_ARRAY(hz)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(ex);
  POLYBENCH_FREE_ARRAY(ey);
  POLYBENCH_FREE_ARRAY(hz);
  POLYBENCH_FREE_ARRAY(_fict_);
  POLYBENCH_FREE_ARRAY(ex_mkl);
  POLYBENCH_FREE_ARRAY(ey_mkl);
  POLYBENCH_FREE_ARRAY(hz_mkl);

  return 0;
}
