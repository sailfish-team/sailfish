// Converts STL data into a dense numpy array suitable for setting geometry
// in a Sailfish simulation.

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>

#include <cvmlcpp/base/Matrix>
#include <cvmlcpp/volume/Geometry>
#include <cvmlcpp/volume/VolumeIO>
#include <cvmlcpp/volume/Voxelizer>

#include "io.hpp"
#include "subdomain.hpp"

using namespace cvmlcpp;
using namespace std;

// Outputs VTK 3.0 UNSTRUCTURED_GRID.
void outputVTK(Matrix<char, 3u> &voxels, const char* filename) {
	std::ofstream output(filename);
	int N = count_if(voxels.begin(), voxels.end(), bind2nd(equal_to<int>(),1));

	output << "# vtk DataFile Version 3.0\nSailfish voxelizer vtk output\nASCII\n"
		<< "DATASET UNSTRUCTURED_GRID\nPOINTS " << N << " float\n";

	int i = 0;
	for (std::size_t x = 0u; x < voxels.extents()[X]; ++x) {
		for (std::size_t y = 0u; y < voxels.extents()[Y]; ++y) {
			for (std::size_t z = 0u; z < voxels.extents()[Z]; ++z) {
				if (voxels[x][y][z] == 1) {
					output << x << " " << y << " " << z << " ";
					if (i++ == 2) {
						i = 0;
						output << std::endl;
					}
				}
			}
		}
	}
	output.close();
}

// TODO: tranform this program into a Python module so that STL geometry can
// be used to directly initialize a simulation
//
// TODO: consider using the distances() function to provide an orientation
// for the walls

int main(int argc, char **argv) {
	Matrix<char, 3u> voxels;
	Geometry<float> geometry;

	double voxel_size = 1.0 / 200.0;

	if (argc < 3) {
		cerr << "Usage: ./voxelizer <STL file> <output_base> [voxel_size]" << endl;
		return -1;
	}

	std::string output_fname(argv[2]);

	if (argc >= 4) {
		voxel_size = atof(argv[3]);
	}

	readSTL(geometry, argv[1]);
  auto orig_geometry = geometry;

	// Scale so that the voxel_size parameter can have a geometry-independent
	// meaning.
	geometry.scaleTo(1.0);

	voxelize(geometry, voxels, voxel_size, 1 /* pad */, kFluid /* inside */, kWall /*outside */);

	const int fluid = count(voxels.begin(), voxels.end(), 0);
	std::cout << "Nodes total: " << voxels.size() << " active: "
		<< round(fluid / (double)voxels.size() * 10000) / 100.0 << "%" << std::endl;

	const std::size_t *ext = voxels.extents();
	std::cout << "Lattice size: " << ext[0] << " " << ext[1] << " " << ext[2] << std::endl;

	SaveAsNumpy(voxels, output_fname);
  SaveConfigFile(geometry, voxels, output_fname);

  // Export a VTK file with the voxelized geometry.
	// outputVTK(voxels, (output_fname + ".vtk").c_str());

	return 0;
}
