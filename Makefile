PROJID = proj1D

LANG = C

LINK_FORTRAN = -lgfortran -no-multibyte-chars
LINK_OPENMP_GCC = -fopenmp
LINK_MKL_GCC = -L/opt/intel/Compiler/11.1/059/mkl/lib/em64t/ \
               -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -liomp5 -lpthread -lm

LINK_OPENMP_ICC = -openmp
LINK_MKL_ICC = -L/opt/intel/Compiler/11.1/059/mkl/lib/em64t/ \
	-Wl,-R/opt/intel/Compiler/11.1/059/mkl/lib/em64t/  -lmkl_lapack \
	-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm


CC = icc
CFLAGS = -O2 -lm -lrt $(LINK_FORTRAN) $(LINK_MKL_ICC) $(LINK_OPENMP_ICC)

FC = ifort
FFLAGS = -O2 $(MKL_ICC) $(OPENMP_ICC)

%.o : %.c
	$(CC) -c $< $(CFLAGS) 

%.o : %.f90
	$(FC) -c $< $(FFLAGS) 

TARGETS =
TARGETS += benchmark--naive
TARGETS += benchmark--blocked
TARGETS += benchmark--blocked_copy

all: $(TARGETS)

common_OBJS = benchmark.o square_dgemm.o stopwatch.o matrix_utils.o 
common_HEADERS = stopwatch.h matrix_utils.h matmult.h

ifeq ($(LANG),C)
naive_MM = matmult_naive.o
blocked_MM = matmult_blocked.o
blocked_copy_MM = matmult_blocked_copy.o
else
naive_MM = matmult_naive.o matmult_wrapper.o
blocked_MM = matmult_blocked.o matmult_wrapper.o
blocked_copy_MM = matmult_blocked_copy.o matmult_wrapper.o
endif

benchmark--naive :  $(common_OBJS) $(naive_MM) 
	$(CC) -o $@ $^ $(CFLAGS) 

benchmark--blocked :  $(common_OBJS) $(blocked_MM) 
	$(CC) -o $@ $^ $(CFLAGS) 

benchmark--blocked_copy :  $(common_OBJS) $(blocked_copy_MM) 
	$(CC) -o $@ $^ $(CFLAGS) 

run-benchmark : benchmark--blocked_copy
	./benchmark--blocked_copy

qsub-benchmark : benchmark--blocked_copy clean-pbs
	qsub benchmark.pbs


DISTFILES = Makefile
DISTFILES += benchmark.c
DISTFILES += benchmark.pbs
DISTFILES += matmult.c
DISTFILES += matmult.f90
DISTFILES += matmult.h
DISTFILES += matmult_blocked.c
DISTFILES += matmult_blocked_copy.c
DISTFILES += matmult_naive.c
DISTFILES += matmult_wrapper.c
DISTFILES += matrix_utils.c
DISTFILES += matrix_utils.h
DISTFILES += square_dgemm.c
DISTFILES += square_dgemm.h
DISTFILES += stopwatch.c
DISTFILES += stopwatch.h

dist: $(PROJID).tgz

$(PROJID).tgz: $(DISTFILES)
	if test -d "$(PROJID)" ; then rm -rf $(PROJID)/ ; fi
	mkdir -p $(PROJID)
	cp -r $(DISTFILES) $(PROJID)
	tar cvf - $(PROJID)/* | gzip -9c > $@

clean-dist:
	rm -rf $(PROJID)/ $(PROJID).tgz

.PHONY : clean
.PHONY : clean-pbs
.PHONY : clean-dist
	
clean : clean-pbs clean-dist
	rm -f $(TARGETS)
	rm -f *.o
	rm -f turnin.tar.gz

clean-pbs : 
	@ [ -d archive ] || mkdir -p ./archive/
	if [ -f Proj1.e* ]; then mv -f Proj1.e* ./archive/; fi;
	if [ -f Proj1.o* ]; then mv -f Proj1.o* ./archive/; fi;

turnin : $(TURNIN_FILES)
	tar czvf turnin.tar.gz *

# eof
