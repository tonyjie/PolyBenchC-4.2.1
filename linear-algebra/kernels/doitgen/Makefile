doitgen: doitgen.c
	gcc -O3 -I../../../utilities -I${MKLROOT}/include -DLARGE_DATASET doitgen.c ../../../utilities/polybench.c -DPOLYBENCH_TIME -o doitgen_mkl -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

clean:
	rm -f doitgen_mkl 