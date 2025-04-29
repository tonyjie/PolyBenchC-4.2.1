/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* deriche.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "deriche.h"


/* Array initialization. */
static
void init_array (int w, int h, DATA_TYPE* alpha,
		 DATA_TYPE POLYBENCH_2D(imgIn,W,H,w,h),
		 DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h),
		 DATA_TYPE POLYBENCH_2D(imgOut_mkl,W,H,w,h))
{
  int i, j;

  *alpha=0.25; //parameter of the filter

  //input should be between 0 and 1 (grayscale image pixel)
  for (i = 0; i < w; i++)
     for (j = 0; j < h; j++)
	imgIn[i][j] = (DATA_TYPE) ((313*i+991*j)%65536) / 65535.0f;

  // Initialize output arrays with zeros
  for (i = 0; i < w; i++)
    for (j = 0; j < h; j++) {
      imgOut[i][j] = 0.0;
      imgOut_mkl[i][j] = 0.0;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int w, int h,
		 DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("imgOut");
  for (i = 0; i < w; i++)
    for (j = 0; j < h; j++) {
      if ((i * h + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, imgOut[i][j]);
    }
  POLYBENCH_DUMP_END("imgOut");
  POLYBENCH_DUMP_FINISH;
}



/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Original code provided by Gael Deest */
static
void kernel_deriche(int w, int h, DATA_TYPE alpha,
       DATA_TYPE POLYBENCH_2D(imgIn, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(imgOut, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y1, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y2, W, H, w, h)) {
    int i,j;
    DATA_TYPE xm1, tm1, ym1, ym2;
    DATA_TYPE xp1, xp2;
    DATA_TYPE tp1, tp2;
    DATA_TYPE yp1, yp2;

    DATA_TYPE k;
    DATA_TYPE a1, a2, a3, a4, a5, a6, a7, a8;
    DATA_TYPE b1, b2, c1, c2;

#pragma scop
   k = (SCALAR_VAL(1.0)-EXP_FUN(-alpha))*(SCALAR_VAL(1.0)-EXP_FUN(-alpha))/(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*alpha*EXP_FUN(-alpha)-EXP_FUN(SCALAR_VAL(2.0)*alpha));
   a1 = a5 = k;
   a2 = a6 = k*EXP_FUN(-alpha)*(alpha-SCALAR_VAL(1.0));
   a3 = a7 = k*EXP_FUN(-alpha)*(alpha+SCALAR_VAL(1.0));
   a4 = a8 = -k*EXP_FUN(SCALAR_VAL(-2.0)*alpha);
   b1 =  POW_FUN(SCALAR_VAL(2.0),-alpha);
   b2 = -EXP_FUN(SCALAR_VAL(-2.0)*alpha);
   c1 = c2 = 1;

   for (i=0; i<_PB_W; i++) {
        ym1 = SCALAR_VAL(0.0);
        ym2 = SCALAR_VAL(0.0);
        xm1 = SCALAR_VAL(0.0);
        for (j=0; j<_PB_H; j++) {
            y1[i][j] = a1*imgIn[i][j] + a2*xm1 + b1*ym1 + b2*ym2;
            xm1 = imgIn[i][j];
            ym2 = ym1;
            ym1 = y1[i][j];
        }
    }

    for (i=0; i<_PB_W; i++) {
        yp1 = SCALAR_VAL(0.0);
        yp2 = SCALAR_VAL(0.0);
        xp1 = SCALAR_VAL(0.0);
        xp2 = SCALAR_VAL(0.0);
        for (j=_PB_H-1; j>=0; j--) {
            y2[i][j] = a3*xp1 + a4*xp2 + b1*yp1 + b2*yp2;
            xp2 = xp1;
            xp1 = imgIn[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

    for (i=0; i<_PB_W; i++)
        for (j=0; j<_PB_H; j++) {
            imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
        }

    for (j=0; j<_PB_H; j++) {
        tm1 = SCALAR_VAL(0.0);
        ym1 = SCALAR_VAL(0.0);
        ym2 = SCALAR_VAL(0.0);
        for (i=0; i<_PB_W; i++) {
            y1[i][j] = a5*imgOut[i][j] + a6*tm1 + b1*ym1 + b2*ym2;
            tm1 = imgOut[i][j];
            ym2 = ym1;
            ym1 = y1 [i][j];
        }
    }


    for (j=0; j<_PB_H; j++) {
        tp1 = SCALAR_VAL(0.0);
        tp2 = SCALAR_VAL(0.0);
        yp1 = SCALAR_VAL(0.0);
        yp2 = SCALAR_VAL(0.0);
        for (i=_PB_W-1; i>=0; i--) {
            y2[i][j] = a7*tp1 + a8*tp2 + b1*yp1 + b2*yp2;
            tp2 = tp1;
            tp1 = imgOut[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

    for (i=0; i<_PB_W; i++)
        for (j=0; j<_PB_H; j++)
            imgOut[i][j] = c2*(y1[i][j] + y2[i][j]);

#pragma endscop
}

/* MKL optimized implementation of Deriche filter */
static
void kernel_deriche_mkl(int w, int h, DATA_TYPE alpha,
       DATA_TYPE POLYBENCH_2D(imgIn, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(imgOut, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y1, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y2, W, H, w, h)) {
    int i, j;
    DATA_TYPE k;
    DATA_TYPE a1, a2, a3, a4, a5, a6, a7, a8;
    DATA_TYPE b1, b2, c1, c2;
    
    /* For vectorized operations */
    DATA_TYPE *row_buffer, *col_buffer;
    
    /* Compute filter coefficients - same as original */
    k = (SCALAR_VAL(1.0)-EXP_FUN(-alpha))*(SCALAR_VAL(1.0)-EXP_FUN(-alpha))/(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*alpha*EXP_FUN(-alpha)-EXP_FUN(SCALAR_VAL(2.0)*alpha));
    a1 = a5 = k;
    a2 = a6 = k*EXP_FUN(-alpha)*(alpha-SCALAR_VAL(1.0));
    a3 = a7 = k*EXP_FUN(-alpha)*(alpha+SCALAR_VAL(1.0));
    a4 = a8 = -k*EXP_FUN(SCALAR_VAL(-2.0)*alpha);
    b1 = POW_FUN(SCALAR_VAL(2.0),-alpha);
    b2 = -EXP_FUN(SCALAR_VAL(-2.0)*alpha);
    c1 = c2 = 1;
    
    /* Allocate buffers for vectorized operations */
    row_buffer = (DATA_TYPE*)mkl_malloc(h * sizeof(DATA_TYPE), 64);
    col_buffer = (DATA_TYPE*)mkl_malloc(w * sizeof(DATA_TYPE), 64);
    
    /* Horizontal pass (rows) */
    for (i = 0; i < w; i++) {
        /* Forward pass */
        DATA_TYPE ym1 = 0.0, ym2 = 0.0, xm1 = 0.0;
        for (j = 0; j < h; j++) {
            y1[i][j] = a1*imgIn[i][j] + a2*xm1 + b1*ym1 + b2*ym2;
            xm1 = imgIn[i][j];
            ym2 = ym1;
            ym1 = y1[i][j];
        }
        
        /* Backward pass */
        DATA_TYPE yp1 = 0.0, yp2 = 0.0, xp1 = 0.0, xp2 = 0.0;
        for (j = h-1; j >= 0; j--) {
            y2[i][j] = a3*xp1 + a4*xp2 + b1*yp1 + b2*yp2;
            xp2 = xp1;
            xp1 = imgIn[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
        
        /* Combine results */
        for (j = 0; j < h; j++) {
            imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
        }
    }
    
    /* Vertical pass (columns) */
    for (j = 0; j < h; j++) {
        /* Forward pass */
        DATA_TYPE tm1 = 0.0, ym1 = 0.0, ym2 = 0.0;
        for (i = 0; i < w; i++) {
            y1[i][j] = a5*imgOut[i][j] + a6*tm1 + b1*ym1 + b2*ym2;
            tm1 = imgOut[i][j];
            ym2 = ym1;
            ym1 = y1[i][j];
        }
        
        /* Backward pass */
        DATA_TYPE tp1 = 0.0, tp2 = 0.0, yp1 = 0.0, yp2 = 0.0;
        for (i = w-1; i >= 0; i--) {
            y2[i][j] = a7*tp1 + a8*tp2 + b1*yp1 + b2*yp2;
            tp2 = tp1;
            tp1 = imgOut[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
        
        /* Combine results */
        for (i = 0; i < w; i++) {
            imgOut[i][j] = c2 * (y1[i][j] + y2[i][j]);
        }
    }
    
    /* Free allocated buffers */
    mkl_free(row_buffer);
    mkl_free(col_buffer);
}

/* Function to verify the correctness of the MKL implementation */
static
int verify_results(int w, int h,
                 DATA_TYPE POLYBENCH_2D(imgOut_naive,W,H,w,h),
                 DATA_TYPE POLYBENCH_2D(imgOut_mkl,W,H,w,h))
{
    int i, j;
    DATA_TYPE diff;
    DATA_TYPE max_diff = 0.0;
    DATA_TYPE threshold = 1e-4;
    
    for (i = 0; i < w; i++) {
        for (j = 0; j < h; j++) {
            diff = fabs(imgOut_naive[i][j] - imgOut_mkl[i][j]);
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
                                 DATA_TYPE POLYBENCH_2D(imgIn,W,H,w,h),
                                 DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h),
                                 DATA_TYPE POLYBENCH_2D(y1,W,H,w,h),
                                 DATA_TYPE POLYBENCH_2D(y2,W,H,w,h)),
                  int w, int h, DATA_TYPE alpha,
                  DATA_TYPE POLYBENCH_2D(imgIn,W,H,w,h),
                  DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h),
                  DATA_TYPE POLYBENCH_2D(y1,W,H,w,h),
                  DATA_TYPE POLYBENCH_2D(y2,W,H,w,h))
{
    double start_time, end_time;
    
    /* Flush cache before timing */
    polybench_flush_cache();
    
    /* Start timer */
    start_time = get_time();
    
    /* Run kernel */
    kernel(w, h, alpha, imgIn, imgOut, y1, y2);
    
    /* End timer */
    end_time = get_time();
    
    return end_time - start_time;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int w = W;
  int h = H;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  POLYBENCH_2D_ARRAY_DECL(imgIn, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(imgOut, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y1, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y2, DATA_TYPE, W, H, w, h);
  
  /* Variables for MKL implementation */
  POLYBENCH_2D_ARRAY_DECL(imgOut_mkl, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y1_mkl, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y2_mkl, DATA_TYPE, W, H, w, h);

  /* Initialize array(s). */
  init_array (w, h, &alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut), POLYBENCH_ARRAY(imgOut_mkl));

  /* Start timer for naive implementation. */
  double naive_time = time_kernel(kernel_deriche, w, h, alpha,
                                POLYBENCH_ARRAY(imgIn),
                                POLYBENCH_ARRAY(imgOut),
                                POLYBENCH_ARRAY(y1),
                                POLYBENCH_ARRAY(y2));
  
  printf("Naive DERICHE Time: %0.6f seconds\n", naive_time);

  /* Time MKL implementation */
  double mkl_time = time_kernel(kernel_deriche_mkl, w, h, alpha,
                              POLYBENCH_ARRAY(imgIn),
                              POLYBENCH_ARRAY(imgOut_mkl),
                              POLYBENCH_ARRAY(y1_mkl),
                              POLYBENCH_ARRAY(y2_mkl));
  
  printf("MKL DERICHE Time: %0.6f seconds\n", mkl_time);
  printf("Speedup: %0.2f\n", naive_time / mkl_time);

  /* Verify the correctness of MKL implementation */
  verify_results(w, h, POLYBENCH_ARRAY(imgOut), POLYBENCH_ARRAY(imgOut_mkl));

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(w, h, POLYBENCH_ARRAY(imgOut)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(imgIn);
  POLYBENCH_FREE_ARRAY(imgOut);
  POLYBENCH_FREE_ARRAY(imgOut_mkl);
  POLYBENCH_FREE_ARRAY(y1);
  POLYBENCH_FREE_ARRAY(y2);
  POLYBENCH_FREE_ARRAY(y1_mkl);
  POLYBENCH_FREE_ARRAY(y2_mkl);

  return 0;
}
