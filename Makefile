CXXFLAGS=-O2
CXX=icc

all: lbm_cuda lbm

%.o: %.c vis.h sim.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

lbm: lbm.cpp sim.o sdl.o vis.h sim.h
	$(CXX) $(CXXFLAGS) lbm.cpp sdl.o sim.o -o lbm -lm -lSDL -lrt -lSDL_ttf

lbm_cuda: lbm.cu sdl.o sim.o vis.h sim.h
	nvcc lbm.cu sdl.cpp sim.cpp --use_fast_math -o lbm_cuda -lSDL -lrt -lSDL_ttf

