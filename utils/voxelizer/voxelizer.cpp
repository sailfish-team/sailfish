#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>

#include <cvmlcpp/base/Matrix>
#include <cvmlcpp/volume/Geometry>
#include <cvmlcpp/volume/VolumeIO>
#include <cvmlcpp/volume/Voxelizer>

using namespace cvmlcpp;
using namespace std;

// Outputs VTK 3.0 UNSTRUCTURED_GRID.
void outputVTK(Matrix<char, 3u> &voxels, const char* filename)
{
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

int main(int argc, char **argv)
{
	Matrix<char, 3u> voxels;
	Geometry<float> geometry;

	double voxel_size = 1.0 / 200.0;

	if (argc < 2) {
		cerr << "Usage: ./voxelizer <STL file> [voxel_size]" << endl;
		return -1;
	}

	if (argc >= 3) {
		voxel_size = atof(argv[2]);
	}

	readSTL(geometry, argv[1]);

	std::cout << "Original bounding box: "
	       << geometry.min(0) << ":" << geometry.max(0) << " "
	       << geometry.min(1) << ":" << geometry.max(1) << " "
	       << geometry.min(2) << ":" << geometry.max(2) << std::endl;
	// Start saving a config file in JSON. This config file can later be used
	// to generate VTK data in the original coordinate system.
	std::ofstream config("output.config");
	config << "{\"bounding_box\": ["
		<< "[" << geometry.min(0) << ", " << geometry.max(0) << "], "
		<< "[" << geometry.min(1) << ", " << geometry.max(1) << "], "
		<< "[" << geometry.min(2) << ", " << geometry.max(2) << "]],";

	// Scale so that the voxel_size parameter can have a geometry-independent
	// meaning.
	geometry.scaleTo(1.0);

	voxelize(geometry, voxels, voxel_size, 1 /* pad */, (char)0 /* inside */, (char)1 /*outside */);

	int fluid = count(voxels.begin(), voxels.end(), 0);
	std::cout << "Nodes total: " << voxels.size() << " active: "
		<< round(fluid / (double)voxels.size() * 10000) / 100.0 << "%" << std::endl;

	const std::size_t *ext = voxels.extents();
	std::cout << "Lattice size: " << ext[0] << " " << ext[1] << " " << ext[2] << std::endl;
	config << "\"size\": [" << ext[0] << ", " << ext[1] << ", " << ext[2] << "]}";
	config.close();

	std::ofstream out("output.npy");
	out << "\x93NUMPY\x01";

	char buf[128] = {0};

	out.write(buf, 1);

	snprintf(buf, 128, "{'descr': 'bool', 'fortran_order': False, 'shape': (%lu, %lu, %lu)}",
			ext[0], ext[1], ext[2]);

	int i, len = strlen(buf);
	unsigned short int dlen = (((len + 10) / 16) + 1) * 16;

	for (i = 0; i < dlen - 10 - len; i++) {
		buf[len+i] = ' ';
	}
	buf[len+i] = 0x0;
	dlen -= 10;

	out.write((char*)&dlen, 2);
	out << buf;

	out.write(&(voxels.begin()[0]), voxels.size());
	out.close();

	// Export a VTK file with the voxelized geometry.
	outputVTK(voxels, "output.vtk");

	return 0;
}
