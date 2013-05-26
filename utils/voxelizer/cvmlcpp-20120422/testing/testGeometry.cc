/***************************************************************************
 *   Copyright (C) 2007 by BEEKHOF, Fokko                                  *
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
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include <cvmlcpp/volume/Geometry>
#include <cvmlcpp/volume/VolumeIO>
#include <iostream>

using namespace cvmlcpp;
using namespace std;

typedef float	ftype;
typedef Geometry<ftype> Geo;

typedef Geo::point_type point_type;
typedef Geo::facet_type facet_type;
typedef Geo::const_facet_iterator FIT;

int main()
{
	Geo g1;

	const std::size_t a = g1.addPoint(1, 1, 1);
	const std::size_t b = g1.addPoint(1, 1, 2);
	const std::size_t c = g1.addPoint(1, 3, 1);
	const std::size_t d = g1.addPoint(4, 3, 1);
	assert(g1.nrPoints() == 4u);
// std::cout << "ABCD " << a << " " << b << " " << c << " " << d << std::endl;
	const std::size_t f1 = g1.addFacet(a, b, c);
	assert(g1.nrFacets() == 1u);
	const std::size_t f2 = g1.addFacet(a, b, d);
	assert(g1.nrFacets() == 2u);
// std::cout << "facets: " << f1 << " " << f2 << std::endl;

	assert(g1.updatePoint(b, 5, 7, 8));

	assert(!g1.erasePoint(a)); // Still in facet f1 & f2
	assert(g1.eraseFacet(f1));
	assert(g1.nrFacets() == 1u);
	assert(!g1.erasePoint(a)); // Still in facet f2
	assert(g1.nrPoints() == 4u);

	assert(g1.eraseFacet(f2));
	assert(g1.erasePoint(a));
	assert(g1.nrPoints() == 3u);

	const std::size_t f3 = g1.addFacet(b, c, d);
// std::cout << "facet 3: " << f3 << std::endl;

	Geo g2(g1);
	assert(g2 == g1);
	const std::size_t e = g2.addPoint(1, 1, 1);
	assert(!g1.updateFacet(f3, e, b, c)); // old one does not have "e"
	assert(g2.updateFacet(f3, e, b, c));
	assert(g2 != g1);

	g1 = g2;
	assert(g2 == g1);

	assert(cvmlcpp::readSTL(g1, "cube.stl"));
	for (int i = 0; i < 3; ++i)
	{
		assert(g1.min(i) < std::numeric_limits<ftype>::max());
		assert(g1.max(i) > std::numeric_limits<ftype>::min());
		assert(g1.min(i) < g1.max(i));
	}
	assert(g1.nrPoints() == 8u);
	g1.scaleTo();
	g1.center();
	g1.rotate(X, 3.0);

	assert(cvmlcpp::readSTL(g1, "d4.stl"));
	assert(g1.nrPoints() == 4u);
	assert(g1.nrFacets() == 4u);
	assert(cvmlcpp::writeSTL(g1, "/tmp/d4bin.stl"));

	return 0;
}

