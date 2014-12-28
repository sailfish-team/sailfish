all: voxelizer voxelizer_octree

CXX=clang++
CXXFLAGS=-stdlib=libc++ -I/opt/local/include

GTEST_INCLUDE=-Igtest/include
GTEST_LIBS=gtest/lib/.libs/libgtest.a gtest/lib/.libs/libgtest_main.a

# Dynamic:
# GTEST_LIBS=-lgtest -lgtest_main

.PHONY: voxelizer voxelizer_octree

voxelizer: voxelizer.cpp
	$(CXX) -O3 -DNDEBUG -std=c++11 $(CXXFLAGS) voxelizer.cpp subdomain.cpp -I. -Icvmlcpp -o $@

voxelizer_octree: voxelizer_octree.cpp
	$(CXX) -O3 -DNDEBUG -std=c++11 $(CXXFLAGS) voxelizer_octree.cpp subdomain.cpp -I. -Icvmlcpp -o $@

subdomain_test: subdomain_test.cpp
	$(CXX) -std=c++11 $(CXXFLAGS) $(GTEST_INCLUDE) subdomain_test.cpp subdomain.cpp -I. -Icvmlcpp $(GTEST_LIBS) -o $@
