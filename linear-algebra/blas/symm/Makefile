symm: symm.c
	gcc -O3 -I../../../utilities -I${MKLROOT}/include -DLARGE_DATASET symm.c ../../../utilities/polybench.c -DPOLYBENCH_TIME -o symm_mkl -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

clean:
	rm -f symm_mkl 