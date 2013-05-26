#!/bin/bash

export OMP_NUM_THREADS=2

make -C .. clean
make -C .. CNF=$1 MODE=debug

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH;.."

for i in test*cc
do
	echo TESTING: $i
	rm -f ./a.out
	case "$1" in
		'gcc')
			g++ \
				-Wall -L.. -I.. -O2 -g -DUSE_THREADS -fopenmp -std=c++0x -funroll-loops -march=native \
				$i -lfftw3 -lfftw3_threads -lfftw3f -lfftw3f_threads -lfftw3l -lfftw3l_threads -lboost_system -lboost_filesystem -lcvmlcpp
		;;

		'icc')
#			icc -I.. -ipo -O3 -g -DUSE_THREADS -DUSE_FFT_FLOAT -DUSE_FFT_LONG -openmp -funroll-loops -axN  -L../lib  -lcvmlcpp -lfftw3 -lfftw3_threads -lfftw3f -lfftw3f_threads -lfftw3l -lfftw3l_threads $i

			icc -I.. -ipo -O3 -g -DUSE_THREADS \
				-openmp -funroll-loops -axN \
				-L.. -lcvmlcpp -lfftw3 -lfftw3_threads \
				-lfftw3f -lfftw3f_threads -lboost_filesystem $i
		;;

		'open64')
			openCC -Wall -I.. -O2 -g -DUSE_THREADS -march=auto -apo \
				-mp -lfftw3 \
				-lfftw3_threads -lfftw3f -lfftw3f_threads \
				-lfftw3l -lfftw3l_threads -lpthread -L.. \
				-lcvmlcpp -lboost_filesystem $i
		;;

		'clang')
			clang -Wall -I.. -O2 -g -DUSE_THREADS \
				-fopenmp -std=c++0x \
				-funroll-loops -march=native -lfftw3 \
				-lfftw3_threads -lfftw3f -lfftw3f_threads \
				-lfftw3l -lfftw3l_threads -lpthread -L.. \
				-lcvmlcpp -lboost_filesystem $i
		;;

		'suncc')
			sunCC -library=stlport4 -features=tmplife \
				-features=tmplrefstatic  -Qoption ccfe \
				-complextmplexp -native -I.. -g -DUSE_THREADS \
				-fast -lfftw3 \
				-lfftw3_threads -lfftw3f -lfftw3f_threads \
				-lfftw3l -lfftw3l_threads -lpthread -L.. \
				-lcvmlcpp -lboost_filesystem $i
		;;

		'watcom')
			owcc -x c++ -I.. -mtune=686 -fptune=686 -Wall -Wc,-xs \
				-O3 -fsigned-char -funroll-loops -mthreads \
				-frerun-optimizer -lfftw3 -lfftw3_threads \
				-lfftw3f -lfftw3f_threads -lfftw3l \
				-lfftw3l_threads -lpthread -lboost_filesystem -L.. -lcvmlcpp $i
		;;

		*)
			${CXX} ${CXXFLAGS} -lfftw3 -lfftw3_threads \
                                -lfftw3f -lfftw3f_threads -lfftw3l \
                                -lfftw3l_threads -lpthread -lboost_filesystem -L.. -lcvmlcpp $i

#			echo Specify "gcc", "icc", "open64" or "clang"
#			echo Options "suncc" and "watcom" are available but these compilers are broken
		;;

	esac

	#valgrind -q ./a.out
	./a.out

done
rm -f ./a.out

# separate testing for CUFFT
#g++ -Wall -I.. -I/usr/local/cuda/include -g -fopenmp  -march=native -DUSE_CUFFT -L /usr/local/cuda/lib64 -lcufft -lcudart testFourier.cc
#./a.out
#rm -f ./a.out
