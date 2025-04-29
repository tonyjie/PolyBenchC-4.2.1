gemver: gemver.c
	gcc -O3 -I../../../utilities -I${MKLROOT}/include -DLARGE_DATASET gemver.c ../../../utilities/polybench.c -DPOLYBENCH_TIME -o gemver_mkl -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

clean:
	rm gemver_mkl