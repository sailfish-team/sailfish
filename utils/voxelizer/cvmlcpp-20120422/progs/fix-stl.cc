/***************************************************************************
 *   Copyright (C) 2011 by F. P. Beekhof                                   *
 *   fpbeekhof@gmail.com                                                   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with program; if not, write to the                              *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include <string>
#include <iostream>
#include <cstdlib>
#include <cvmlcpp/volume/VolumeIO>
#include <cvmlcpp/volume/Voxelizer>
#include <cvmlcpp/volume/SurfaceExtractor>

void usage(const char * const name)
{
	std::cout << "usage: "<<name<< " [-a] <stl-file> <sample-size> <output-stl-file>\n\n"
		<< " -a : write ascii STL file\n" << std::endl;
	exit(0);
}

int main(int argc, const char * const argv[])
{
	using namespace cvmlcpp;

	typedef float T;

	if (argc < 4 || argc > 5)
		usage(argv[0]);

	const bool binary = (argc == 4) && std::string(argv[1]) == "-a";
	const int arg1 = binary ? 3 : 2;
	

	const T voxel_size = std::atof(argv[arg1+1]);
	if (!(voxel_size > 0))
	{
		std::cout << "ERROR: voxel size found as "<<voxel_size<<
		", should be positive float value. Bailing out.\n"<<std::endl;
		usage(argv[0]);
		return -1;
	}

	Geometry<T> geometry;
	if (!readSTL(geometry, argv[arg1]))
	{
		std::cout << "ERROR: can't read STL file ["<<argv[1]<<"], bailing out.\n"<<std::endl;
		return -1;
	}

	Matrix<char, 3> voxels;
	if (!voxelize(geometry, voxels, voxel_size))
	{
		std::cout << "ERROR: voxelization failed, bailing out.\n"<<std::endl;
		return -1;
	}

	extractSurface(voxels, geometry);

	if (!writeSTL(geometry, argv[arg1+2]))
	{
		std::cout << "ERROR: can't write STL file ["<<argv[arg1+2]<<"], bailing out.\n"<<std::endl;
		return -1;
	}

	return 0;
}
