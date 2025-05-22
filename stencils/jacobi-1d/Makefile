SRC = jacobi-1d

naive-o3: ${SRC}.c
	gcc -O3 -I../../utilities -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE ${SRC}.c ../../utilities/polybench.c -DPOLYBENCH_TIME -march=native -o naive-o3 

optimized: ${SRC}.c
	gcc -O3 -I../../utilities -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE ${SRC}.c ../../utilities/polybench.c -DPOLYBENCH_TIME -march=native -fopenmp -ftree-vectorize -ffast-math -DMKL_ILP64 -m64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -o optimized

pluto: ${SRC}.c
	polycc ${SRC}.c --tile -o ${SRC}_tiled.c
	gcc -O3 -I../../utilities -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE ${SRC}_tiled.c ../../utilities/polybench.c -DPOLYBENCH_TIME -march=native -o pluto

HALIDE_PATH = /home/jl4257/anaconda3/envs/autogen/lib/python3.11/site-packages/halide
halide: ${SRC}_halide.cpp
	g++ -std=c++17 ${SRC}_halide.cpp -I${HALIDE_PATH}/include -L${HALIDE_PATH}/lib64 \
	-O3 -march=native \
	-DHALIDE_USE_AUTO_SCHEDULER -DAUTO_SCHEDULE_HOST -DHL_TARGET_FEATURE_AVX512 \
	-lHalide -lpthread -ldl \
	-o halide

mkl: ${SRC}.c
	gcc -O3 -I../../utilities -I${MKLROOT}/include -DLARGE_DATASET -DDATA_TYPE_IS_DOUBLE ${SRC}.c ../../utilities/polybench.c -DPOLYBENCH_TIME -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -march=native -o mkl

clean:
	rm -f mkl
	rm -f naive-o3
	rm -f pluto
	rm -f optimized
	rm -f ${SRC}_tiled.c
	rm -f halide

