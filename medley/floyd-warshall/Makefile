floyd-warshall: floyd-warshall.c
	gcc -O3 -I../../utilities -I${MKLROOT}/include -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE floyd-warshall.c ../../utilities/polybench.c -DPOLYBENCH_TIME -o floyd-warshall_mkl -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

clean:
	rm floyd-warshall_mkl 