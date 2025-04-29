atax: atax.c
	gcc -O3 -I../../../utilities -I${MKLROOT}/include -DLARGE_DATASET atax.c ../../../utilities/polybench.c -DPOLYBENCH_TIME -o atax_mkl -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

clean:
	rm -f atax_mkl 