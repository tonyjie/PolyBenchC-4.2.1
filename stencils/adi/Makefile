adi: adi.c
	gcc -O3 -I../../utilities -I${MKLROOT}/include -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE adi.c ../../utilities/polybench.c -DPOLYBENCH_TIME -o adi_mkl -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

clean:
	rm adi_mkl 