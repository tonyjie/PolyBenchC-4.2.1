gemm: gemm.c
	gcc -O3 -I../../../utilities -I${MKLROOT}/include -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE gemm.c ../../../utilities/polybench.c -DPOLYBENCH_TIME -o gemm_mkl -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

clean:
	rm gemm_mkl