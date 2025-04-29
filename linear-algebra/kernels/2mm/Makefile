2mm: 2mm.c
	gcc -O3 -I../../../utilities -I${MKLROOT}/include -DLARGE_DATASET 2mm.c ../../../utilities/polybench.c -DPOLYBENCH_TIME -o 2mm_mkl -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

clean:
	rm -f 2mm_mkl 