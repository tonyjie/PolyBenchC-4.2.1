# Makefile for building all MKL implementations of PolyBench kernels

SUBDIRS_BLAS = linear-algebra/blas/gemm linear-algebra/blas/gemver linear-algebra/blas/gesummv \
               linear-algebra/blas/syr2k linear-algebra/blas/syrk linear-algebra/blas/trmm \
               linear-algebra/blas/symm

SUBDIRS_KERNELS = linear-algebra/kernels/2mm linear-algebra/kernels/3mm linear-algebra/kernels/atax \
                  linear-algebra/kernels/bicg linear-algebra/kernels/doitgen linear-algebra/kernels/mvt

SUBDIRS_SOLVERS = linear-algebra/solvers/cholesky linear-algebra/solvers/durbin \
                  linear-algebra/solvers/gramschmidt linear-algebra/solvers/lu \
                  linear-algebra/solvers/ludcmp linear-algebra/solvers/trisolv

SUBDIRS = $(SUBDIRS_BLAS) $(SUBDIRS_KERNELS) $(SUBDIRS_SOLVERS)

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done 