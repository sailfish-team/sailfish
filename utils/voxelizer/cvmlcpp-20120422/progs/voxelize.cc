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

void usage(const char * const name)
{
	std::cout << "usage: "<<name<< " <stl-file> <voxelsize>" << std::endl;
	exit(0);
}

int main(int argc, const char * const argv[])
{
	using namespace cvmlcpp;

	typedef float T;

	if (argc != 3)
		usage(argv[0]);

	const T voxel_size = std::atof(argv[2]);
	if (!(voxel_size > 0))
	{
		std::cout << "ERROR: voxel size found as "<<voxel_size<<
		", should be positive float value. Bailing out.\n"<<std::endl;
		usage(argv[0]);
		return -1;
	}

	Geometry<T> geometry;
	if (!readSTL(geometry, argv[1]))
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

	const std::string input(argv[1]);
	std::string output;
	if (argc == 5)
		output = argv[4];
	else if (input.length() > 5)
		output = input.substr(0, input.length() - 4); // strip extention
	else
		output = input;
	output += ".dat";

	if (!writeVoxels(voxels, output))
	{
		std::cout << "ERROR: can't write voxel file ["<<output<<"], bailing out.\n"<<std::endl;
		return -1;
	}

	return 0;
}
