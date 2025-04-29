syr2k: syr2k.c
	gcc -O3 -I../../../utilities -I${MKLROOT}/include -DLARGE_DATASET syr2k.c ../../../utilities/polybench.c -DPOLYBENCH_TIME -o syr2k_mkl -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

clean:
	rm -f syr2k_mkl 