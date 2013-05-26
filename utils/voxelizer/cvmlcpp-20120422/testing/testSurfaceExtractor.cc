/***************************************************************************
 *   Copyright (C) 2005, 2006, 2007 by F. P. Beekhof                       *
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

#include <vector>

#include <boost/multi_array.hpp>

#include <cvmlcpp/base/Matrix>
#include <cvmlcpp/volume/Geometry>
#include <cvmlcpp/volume/SurfaceExtractor>

// typedef Matrix<int, 3u> iMatrix3D;
// typedef array_traits<Matrix, int, 3u, std::vector<int> > MTraits;

using namespace cvmlcpp;

typedef boost::multi_array<int, 3u> iMatrix3D;
typedef array_traits<boost::multi_array, int, 3u, std::allocator<int> > MTraits;

typedef Geometry<float> fGeometry;

int main()
{
	iMatrix3D m = MTraits::create({ 12, 13, 14 }, 1);
	fGeometry g;

	extractSurface(m, g);

	return 0;
}
