all:
	@echo "======================================================================"
	@echo "Proj 1: Distribued Matrix Multiply"
	@echo ""
	@echo "Valid build targets:"
	@echo ""
	@echo "unittest_mm_original : Build matrix multiply unittests"
	@echo "  unittest_mm_openmp : Build matrix multiply unittests"
	@echo "     unittest_mm_mkl : Build matrix multiply unittests"
	@echo "unittest_mm_blocking : Build matrix multiply unittests"
	@echo "      unittest_summa : Build summa unittests"
	@echo "    time_mm_original : Build program to time local_mm (original)"
	@echo "      time_mm_openmp : Build program to time local_mm (open MP)"
	@echo "         time_mm_mkl : Build program to time local_mm (Intel MKL)"
	@echo "    time_mm_blocking : Build program to time local_mm (blocking technique)"
	@echo "          time_summa : Build program to time summa"
	@echo "    run--unittest_mm : Submit unittest_mm job"
	@echo " run--unittest_summa : Submit unittest_summa job"
	@echo "        run--time_mm : Submit time_mm job"
	@echo "     run--time_summa : Submit time_summa job"
	@echo "              turnin : Create tarball with answers and results for T-Square"
	@echo "               clean : Removes generated files, junk"
	@echo "           clean-pbs : Removes genereated pbs files"
	@echo "======================================================================"

LANG = C

LINK_OPENMP_GCC = -fopenmp
LINK_MKL_GCC = -L/opt/intel/Compiler/11.1/059/mkl/lib/em64t/ \
	-lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -liomp5 -lpthread



CC = mpicc
CFLAGS_OPENMP = -O -Wall -Wextra -lm $(LINK_OPENMP_GCC)
CFLAGS_MKL = -O -Wall -Wextra -lm $(LINK_MKL_GCC)
CFLAGS = -O -Wall -Wextra -lm

FC = mpif90
FFLAGS = -O $(MKL_GCC) $(OPENMP_GCC)



MM = local_mm.o
SUMMA = summa.o

local_mm_original.o : local_mm.c local_mm.f90 local_mm.h
	$(CC) $(CFLAGS) -o $@ -c local_mm.c

local_mm_openmp.o : local_mm.c local_mm.f90 local_mm.h
	$(CC) $(CFLAGS_OPENMP) -DUSE_OPEN_MP -o $@ -c local_mm.c

local_mm_mkl.o : local_mm.c local_mm.f90 local_mm.h
	$(CC) $(CFLAGS_MKL) -DUSE_MKL -o $@ -c local_mm.c

local_mm_blocking.o : local_mm.c local_mm.f90 local_mm.h
	$(CC) $(CFLAGS) -DUSE_BLOCKING -o $@ -c local_mm.c

matrix_utils.o : matrix_utils.c matrix_utils.h
	$(CC) $(CFLAGS) -o $@ -c $<

unittest_mm_original : unittest_mm.c matrix_utils.o local_mm.o
	$(CC) $(CFLAGS) -o $@ $^

unittest_mm_openmp : unittest_mm.c matrix_utils.o local_mm_openmp.o
	$(CC) $(CFLAGS_OPENMP) -o $@ $^

unittest_mm_mkl : unittest_mm.c matrix_utils.o local_mm_mkl.o
	$(CC) $(CFLAGS_MKL) -o $@ $^

unittest_mm_blocking : unittest_mm.c matrix_utils.o local_mm_blocking.o
	$(CC) $(CFLAGS) -o $@ $^

time_mm_original : time_mm.c matrix_utils.o local_mm_original.o
	$(CC) $(CFLAGS) -o $@ $^

time_mm_openmp : time_mm.c matrix_utils.o local_mm_openmp.o
	$(CC) $(CFLAGS_OPENMP) -o $@ $^

time_mm_mkl : time_mm.c matrix_utils.o local_mm_mkl.o
	$(CC) $(CFLAGS_MKL) -o $@ $^

time_mm_blocking : time_mm.c matrix_utils.o local_mm_blocking.o
	$(CC) $(CFLAGS) -o $@ $^

time_mm : time_mm_original time_mm_openmp time_mm_blocking time_mm_mkl

unittest_mm : unittest_mm_original unittest_mm_blocking unittest_mm_mkl unittest_mm_openmp

unittest_summa : matrix_utils.o $(MM) $(SUMMA) unittest_summa.o
ifeq ($(LANG),C)
	$(CC) $(CFLAGS) -o $@ $^
else
	$(FC) $(FFLAGS) -o $@ $^
endif

time_summa : matrix_utils.o $(MM) $(SUMMA) time_summa.o
ifeq ($(LANG),C)
	$(CC) $(CFLAGS) -o $@ $^
else
	$(FC) $(FFLAGS) -o $@ $^
endif

summa.o : summa.c summa.f90 summa.h local_mm.h
ifeq ($(LANG),C)
	$(CC) $(CFLAGS) -o summa.o -c summa.c
else
	$(FC) $(FFLAGS) -o summa.o -c summa.f90
endif

unittest_summa.o : unittest_summa.c
	$(CC) $(CFLAGS) -o $@ -c $<

time_summa.o : time_summa.c
	$(CC) $(CFLAGS) -o $@ -c $<

summa_wrapper.o : summa_wrapper.c
	$(CC) $(CFLAGS) -o $@ -c $<

local_mm_wrapper.o : local_mm_wrapper.c
	$(CC) $(CFLAGS) -o $@ -c $<


.PHONY : clean
.PHONY : clean-pbs
.PHONY : time_mm
.PHONY : unittest_mm
	
clean : clean-pbs
	rm -f unittest_mm unittest_summa time_summa
	rm -f time_mm_mkl time_mm_openmp time_mm_original time_mm_blocking
	rm -f unittest_mm_original unittest_mm_mkl unittest_mm_openmp unittest_mm_blocking
	rm -f *.o
	rm -f turnin.tar.gz

clean-pbs : 
	@ [ -d archive ] || mkdir -p ./archive/
	if [ -f Proj1.e* ]; then mv -f Proj1.e* ./archive/; fi;
	if [ -f Proj1.o* ]; then mv -f Proj1.o* ./archive/; fi;

run--unittest_mm : unittest_mm clean-pbs
	qsub unittest_mm.pbs

run--unittest_summa : unittest_summa clean-pbs
	qsub unittest_summa.pbs

run--time_mm : time_mm clean-pbs
	qsub time_mm.pbs

run--time_summa : time_summa clean-pbs
	qsub time_summa.pbs

turnin : $(TURNIN_FILES)
	tar czvf turnin.tar.gz *

# eof
