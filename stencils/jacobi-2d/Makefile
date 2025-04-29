jacobi-2d: jacobi-2d.c
	gcc -O3 -I../../utilities -I${MKLROOT}/include -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE jacobi-2d.c ../../utilities/polybench.c -DPOLYBENCH_TIME -o jacobi-2d_mkl -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

clean:
	rm -f jacobi-2d_mkl 