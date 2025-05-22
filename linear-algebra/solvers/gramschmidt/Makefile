naive: gramschmidt.c
	gcc -O3 -I../../../utilities -DLARGE_DATASET gramschmidt.c ../../../utilities/polybench.c -DPOLYBENCH_TIME -o gramschmidt_naive -lm -march=native

pluto: gramschmidt.c
	gcc -O3 -I../../../utilities -DLARGE_DATASET gramschmidt_tiled.c ../../../utilities/polybench.c -DPOLYBENCH_TIME -o gramschmidt_tiled -lm -march=native

generate_tiled_generated: gramschmidt.c
	polycc gramschmidt.c --noparallel -o gramschmidt_tiled_generated.c

llm_optimized: gramschmidt.c
	gcc -O3 -I../../../utilities -DLARGE_DATASET gramschmidt_llm_optimize.c ../../../utilities/polybench.c -DPOLYBENCH_TIME -o gramschmidt_llm_optimized -lm -march=native

mkl: gramschmidt_mkl.c
	gcc -O3 -I../../../utilities -I${MKLROOT}/include -DLARGE_DATASET gramschmidt_mkl.c ../../../utilities/polybench.c -DPOLYBENCH_TIME -o gramschmidt_mkl -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm

clean:
	rm -f gramschmidt_naive
	rm -f gramschmidt_mkl 
	rm -f gramschmidt_tiled
	rm -f gramschmidt_llm_optimized