CXXFLAGS=-O2
CXX=icc

lbm: lbm.cpp
	$(CXX) $(CXXFLAGS) lbm.cpp -o lbm -lm
