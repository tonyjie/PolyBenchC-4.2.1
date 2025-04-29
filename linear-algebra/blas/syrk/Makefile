# Makefile for SYRK MKL implementation

syrk: syrk.c
	gcc -O3 -I../../../utilities -I${MKLROOT}/include -DLARGE_DATASET syrk.c ../../../utilities/polybench.c -DPOLYBENCH_TIME -o syrk_mkl -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

clean:
	rm -f syrk_mkl 