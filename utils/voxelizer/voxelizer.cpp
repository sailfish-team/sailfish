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
void outputVTK(Matrix<char, 3u> &voxels, const char* filename, float length)
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

// TODO: use the 'pad' option in voxelize to generate an additional layer of
// nodes outside of the object being voxelizer
//
// TODO: tranform this program into a Python module so that STL geometry can
// be used to directly initialize a simulation
//
// TODO: consider using the distances() function to provide an orientation
// for the walls

int main(int argc, char **argv)
{
	Matrix<char, 3u> voxels;
	Geometry<float> geometry;

	double resolution = 0.0003;

	if (argc < 2) {
		cerr << "You need to specify an STL file to voxelize." << endl;
		return -1;
	}

	if (argc >= 3) {
		resolution = atof(argv[2]);
	}

	readSTL(geometry, argv[1]);

	voxelize(geometry, voxels, resolution);
	std::cout << voxels.size() << std::endl;

	const std::size_t *ext = voxels.extents();
	std::cout << ext[0] << " " << ext[1] << " " << ext[2] << std::endl;

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

	std::cout << dlen << std::endl;

	out.write((char*)&dlen, 2);
	out << buf;

	out.write(&(voxels.begin()[0]), voxels.size());
	out.close();

	// Export a VTK file with the voxelized geometry.
	outputVTK(voxels, "output.vtk", resolution);

	return 0;
}
