CXXFLAGS=-O2
CXX=icc

all: lbm_cuda lbm

lbm: lbm.cpp
	$(CXX) $(CXXFLAGS) lbm.cpp -o lbm -lm -lhdf5

lbm_cuda: lbm.cu
	nvcc lbm.cu --use_fast_math -o lbm_cuda
