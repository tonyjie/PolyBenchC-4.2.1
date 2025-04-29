3mm: 3mm.c
	gcc -O3 -I../../../utilities -I${MKLROOT}/include -DLARGE_DATASET 3mm.c ../../../utilities/polybench.c -DPOLYBENCH_TIME -o 3mm_mkl -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

clean:
	rm -f 3mm_mkl 