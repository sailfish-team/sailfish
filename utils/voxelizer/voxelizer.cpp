#include <cvmlcpp/base/Matrix>
#include <cvmlcpp/volume/Geometry>
#include <cvmlcpp/volume/VolumeIO>
#include <cvmlcpp/volume/Voxelizer>
#include <fstream>
#include <cstdio>

using namespace cvmlcpp;
using namespace std;

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

	return 0;
}
