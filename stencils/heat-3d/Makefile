heat-3d: heat-3d.c
	gcc -O3 -I../../utilities -I${MKLROOT}/include -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE heat-3d.c ../../utilities/polybench.c -DPOLYBENCH_TIME -o heat-3d_mkl -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

clean:
	rm -f heat-3d_mkl 