CXXFLAGS=-O2
CXX=icc

all: lbm_cuda lbm

lbm: lbm.cpp sim.cpp sdl.cpp
	$(CXX) $(CXXFLAGS) lbm.cpp sdl.cpp sim.cpp -o lbm -lm -lhdf5 -lSDL

lbm_cuda: lbm.cu sdl.cpp sim.cpp vis.h sim.h
	nvcc lbm.cu sdl.cpp sim.cpp --use_fast_math -o lbm_cuda -lSDL

