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

#include <iostream>
#include <fstream>
#include <vector>

#include <cvmlcpp/base/stl_cstdint.h>

#include <cvmlcpp/array/ArrayIO>
#include <cvmlcpp/math/Math>
#include <cvmlcpp/base/Meta>
#include <cvmlcpp/base/StringTools>
#include <omptl/omptl_algorithm>

namespace cvmlcpp
{

namespace detail
{

template <typename T>
bool writeSTLBinary(const Geometry<T> &geometry, std::ofstream &f)
{
	typedef typename Geometry<T>::point_type	point_type;
	typedef typename Geometry<T>::vector_type	vector_type;
	typedef typename Geometry<T>::facet_type	facet_type;

	char buf[80] = "Geometry in binary STL format. http://tech.unige.ch/cvmlcpp/";
	f.write(buf, 80);

	const int32_t nFacets = geometry.nrFacets();
	f.write((char *)&nFacets, sizeof(int32_t));

	for (typename Geometry<T>::const_facet_iterator facet = geometry.facetsBegin();
	     facet != geometry.facetsEnd(); ++facet)
	{
		const char pad [] = "  ";

		const fVector3D normal( (facet->normal()) );

		// Write normal
		f.write((char *)&normal[0], 3*sizeof(float));

		for(int i = 0; i < 3; i++)
		{
			const fPoint3D p( (geometry.point((*facet)[i])) );
			f.write((char *)&p[0], sizeof(float)*3);
		}

		f.write(pad, 2);
	}

	return f.good();
}

template <typename T>
bool readSTLASCII(Geometry<T> &geometry, std::ifstream &f)
{
	typedef typename Geometry<T>::point_type	point_type;
	typedef typename Geometry<T>::vector_type	vector_type;
	typedef typename Geometry<T>::facet_type	facet_type;

	std::vector<Point3D<T> > points;
	std::vector<vector_type>    normals;

	std::string input;

	// Read "solid"
	f >> input; to_lower(input);
	if (input != "solid")
		return false;

	// Read name, if it is there
	std::getline(f, input);

	// Read first facet
	f >> input; to_lower(input);

	//std::size_t pIndex = 0;
	while ((input != "endsolid") && (f.good()))
	{
		if (input != "facet") // We must have read "facet"
		{
			std::cerr << "[facet] or [endsolid] expected" << std::endl;
			return false;
		}

		// read "normal"
		f >> input; to_lower(input);
		if (input != "normal") // We must have read "normal"
		{
			std::cerr << "[normal] expected" << std::endl;
			return false;
		}

		// Read normal
		fVector3D normal;
		f >> normal[X]; f >> normal[Y]; f >> normal[Z];

		// Read "outer loop"
		f >> input; to_lower(input);
		if (input != "outer") // We must have read "outer"
		{
			std::cerr << "[outer] expected" << std::endl;
			return false;
		}
		f >> input; to_lower(input);
		if (input != "loop") // We must have read "loop"
		{
			std::cerr << "[loop] expected" << std::endl;
			return false;
		}

		// Read 3 points
		for (unsigned i = 0u; i < 3u; ++i)
		{
			// Read "vertex"
			f >> input; to_lower(input);
			if (input != "vertex") // We must have read "vertex"
			{
				std::cerr << "[vertex] expected" << std::endl;
				return false;
			}

			fPoint3D pt;
			f >> pt[X]; f >> pt[Y]; f >> pt[Z];
			points.push_back(pt);
		}
		assert(points.size() >= 3);
		assert(points.size() % 3 == 0);

		// Recompute normal if needed
		const float norm = dotProduct(normal, normal);
		if (!(norm > 0.0))
		{
			// recompute
			const unsigned a = points.size() - 3;
			const unsigned b = points.size() - 2;
			const unsigned c = points.size() - 1;
			const vector_type ab = points[b] - points[a];
			const vector_type ac = points[c] - points[a];
 			normal = crossProduct(ab, ac);
			normal /= modulus(normal);
		}

		normals.push_back(normal);

		// Read "std::endloop"
		f >> input; to_lower(input);
		if (input != "endloop") // We must have read "std::endloop"
		{
			std::cerr << "[endloop] expected" << std::endl;
			return false;
		}

		// Read "endfacet"
		f >> input; to_lower(input);
		if (input != "endfacet") // We must have read "endfacet"
		{
			std::cerr << "[endfacet] expected" << std::endl;
			return false;
		}

		// Read either new "vertex" or "endsolid"
		f >> input; to_lower(input);
	}

	if (f.fail())
		return false;

	Geometry<T> g(points.begin(), points.end(), normals.begin());
	using std::swap;
	swap(g, geometry);

	return true;
}

template <typename T>
bool readSTLBinary(Geometry<T> &geometry, std::ifstream &f)
{
	typedef typename Geometry<T>::point_type	point_type;
	typedef typename Geometry<T>::vector_type	vector_type;
	typedef typename Geometry<T>::facet_type	facet_type;

	// Read header
	char comment[80];
	f.read(comment, 80);
	if (!f.good())
		return false;
	comment[79] = 0;

	int32_t nFacets;
	f.read(reinterpret_cast<char *>(&nFacets), sizeof(int32_t));
	if (!f.good())
		return false;

	const unsigned nPoints = 3u * nFacets;
	std::vector<Point3D<T> > points(nPoints);
	std::vector<vector_type> normals(nFacets);

	for (int i = 0; i < nFacets; ++i)
	{
		// Read facet_type data into buffer
		float buf[50 / sizeof(float) + 1];
		f.read(reinterpret_cast<char *>(buf), 50);
		if (f.fail())
			return false;

		// Read 3 idxPoints of facet_type
		for (unsigned j = 0; j < 3; j++)
		{
			const unsigned ptIndex = 3u*i + j;
			std::copy(&buf[3u*j+3u], &buf[3u*j+3u] + 3u,
				  points[ptIndex].begin());

			// Test: double point in facet ?
// 			for (unsigned k = 0; k < j; ++k)
// 				if (points[3u*i+k] == points[ptIndex])
// 					return false;
		}

		// read, normalize and add normal, i.e. first 3 floats in buffer
		normals[i].load(&buf[0], &buf[3]);

		// Recompute normal if needed
		const float norm = dotProduct(normals[i], normals[i]);
		if (!(norm > 0.0))
		{
			// recompute
			const unsigned a = 3*i;
			const unsigned b = 3*i+1;
			const unsigned c = 3*i+2;
			const vector_type ab = points[b] - points[a];
			const vector_type ac = points[c] - points[a];
			normals[i] = crossProduct(ab, ac);
			normals[i] /= modulus(normals[i]);
		}
	}

	geometry.loadGeometry(points.begin(), points.end(), normals.begin());

	return true;
}

} // namespace detail

template <typename T>
bool writeSTL(const Geometry<T> &geometry, const std::string fileName,
 		const bool binary)
{
	bool ok = false;

	if (binary)
	{
		std::ofstream f(fileName.c_str(), std::ios::out|std::ios::binary);
		if (!f.good())
			return false;
		ok = detail::writeSTLBinary(geometry, f);
		f.close();
	}

	return ok;
}

template <typename T>
bool readSTL(Geometry<T> &geometry, const std::string fileName)
{
	std::ifstream f(fileName.c_str(), std::ios::in);
	if (!f.good())
		return false;

	const std::string asciiHeader = "solid";
	char buf[6]; buf[5] = 0;
	f.read(buf, 5);

	bool result = false;

	if (std::string(buf) == asciiHeader)
	{
		f.seekg(0, std::ios::beg);
		if(f.good())
			result = detail::readSTLASCII (geometry, f);
	}
	else
	{
		f.close();
		f.open(fileName.c_str(), std::ios::in|std::ios::binary);
		if(f.good())
			result = detail::readSTLBinary(geometry, f);
	}

	f.close();

	return result;
}

} // namespace cvmlcpp
