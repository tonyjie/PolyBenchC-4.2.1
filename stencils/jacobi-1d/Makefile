jacobi-1d: jacobi-1d.c
	gcc -O3 -I../../utilities -I${MKLROOT}/include -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE jacobi-1d.c ../../utilities/polybench.c -DPOLYBENCH_TIME -o jacobi-1d_mkl -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

clean:
	rm -f jacobi-1d_mkl 