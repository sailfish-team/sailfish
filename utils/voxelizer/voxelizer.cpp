#include <cvmlcpp/base/Matrix>
#include <cvmlcpp/volume/Geometry>
#include <cvmlcpp/volume/VolumeIO>
#include <cvmlcpp/volume/Voxelizer>
#include <fstream>
#include <cstdio>

using namespace cvmlcpp;
using namespace std;

// TODO: tranform this program into a Python module so that STL geometry can
// be used to directly initialize a simulation
//
// TODO: consider using the distances() function to provide an orientation
// for the walls

int main(int argc, char **argv)
{
	Matrix<char, 3u> voxels;
	Geometry<float> geometry;

	double voxel_size = 1.0 / 200.0;;

	if (argc < 2) {
		cerr << "Usage: ./voxelizer <STL file> [voxel_size]" << endl;
		return -1;
	}

	if (argc >= 3) {
		voxel_size = atof(argv[2]);
	}

	readSTL(geometry, argv[1]);

	geometry.scaleTo(1.0);
	std::cout << "Bounding box: "
	       << geometry.max(0) - geometry.min(0) << " "
	       << geometry.max(1) - geometry.min(1) << " "
	       << geometry.max(2) - geometry.min(2) << std::endl;

	voxelize(geometry, voxels, voxel_size, 1 /* pad */, (char)0 /* inside */, (char)1 /*outside */);
	std::cout << "Total nodes: " << voxels.size() << std::endl;

	const std::size_t *ext = voxels.extents();
	std::cout << "Lattice size: " << ext[0] << " " << ext[1]
		<< " " << ext[2] << std::endl;

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

	return 0;
}
