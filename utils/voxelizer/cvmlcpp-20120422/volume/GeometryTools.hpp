/***************************************************************************
 *   Copyright (C) 2005 by BEEKHOF, Fokko                                  *
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

#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>

#include <cvmlcpp/volume/Geometry>

void verify(const Geometry<T> &geometry)
{
	// Test 1: all points in one Facet are different.
	for (std::vector<Facet>::const_iterator f = facets.begin();
						f != facets.end(); ++f)
		if( (f[A] == f[B]) || (f[B] == f->c) || (f->b == f->c) )
		{
			cout << "FileGeometry::verify() Double point in Facet";
			cout << endl;
			return false;
		}

	// Test 2: Each point must be in at least 3 facets to close the surface.
	for (unsigned p = 0; p < points.size(); ++p)
		if( (pointFacetMap[p].size()<3) && (pointFacetMap[p].size()>0) )
		{
			cout << "Not enough facets to close the surface around "
			     << "point:" << endl << "[" << p << "]";
			for (set<unsigned>::iterator si =
						pointFacetMap[p].begin();
					si != pointFacetMap[p].end(); ++si)
				cout << " " << *si;
			cout << endl;

			return false;
		}

	// Test 3: all lines must be connected to exactly 2 facets.
	unsigned inserts = 0;
	std::set<pair<unsigned, unsigned> > lines;
	for (std::vector<Facet>::iterator f = facets.begin();
	     f != facets.end(); ++f)
	{
		lines.insert(pair<unsigned, unsigned>(min(f->a, f->b),
							max(f->a, f->b)));
		lines.insert(pair<unsigned, unsigned>(min(f->a, f->c),
							max(f->a, f->c)));
		lines.insert(pair<unsigned, unsigned>(min(f->b, f->c),
							max(f->b, f->c)));
		inserts += 3;
	}
	// all lines must appear twice - once in every two neighbouring facets.
	// thus, for each line, there must be two facets to close the surface.
	if (inserts % 2 != 0)
	{
		cout << "FileGeometry::verify() uneven nr of lines." << endl;
		return false;
	}

	// Each line of a facet must occur in exactly one other facet.
	for (std::set<pair<unsigned, unsigned> >::iterator lineIt
 			= lines.begin(); lineIt != lines.end(); ++lineIt)
	{
		std::set<unsigned>	sharedFacets;

		std::insert_iterator<set<unsigned> > facIns(sharedFacets,
							sharedFacets.begin());
		std::set_intersection(pointFacetMap[lineIt->first].begin(),
				pointFacetMap[lineIt->first].end(),
				pointFacetMap[lineIt->second].begin(),
				pointFacetMap[lineIt->second].end(),
				facIns);

		if (sharedFacets.size() != 2)
		{

			const unsigned	a	= lineIt->first;
			const unsigned	b	= lineIt->second;

			cout << "Line connected to " <<
				((sharedFacets.size() < 2)?"more":"less") <<
				" than 2 facets." << endl;
			cout << "[" << a << "] ";
			for (std::set<unsigned>::iterator it =
						pointFacetMap[a].begin();
					it != pointFacetMap[a].end(); ++it)
				cout << *it << " ";
			cout << endl << "[" << b << "] ";
			for (std::set<unsigned>::iterator it =
						pointFacetMap[b].begin();
					it != pointFacetMap[b].end(); ++it)
				cout << *it << " ";
			cout << endl;

			return false;
		}
	}
	return true;
}

} // namespace
