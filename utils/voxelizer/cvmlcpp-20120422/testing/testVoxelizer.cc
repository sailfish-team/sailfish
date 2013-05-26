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

#include <cstdlib>
#include <limits>
#include <cvmlcpp/base/Matrix>
#include <cvmlcpp/volume/DTree>
#include <cvmlcpp/volume/Geometry>
#include <cvmlcpp/volume/VolumeIO>
#include <cvmlcpp/volume/Voxelizer>

#include <boost/multi_array.hpp>

using namespace cvmlcpp;

double rnd(const double minval, const double maxval)
{
	assert(minval >= -std::numeric_limits<double>::max());
	assert(maxval <=  std::numeric_limits<double>::max());
	const double val = drand48() * (maxval - minval) + minval;
	assert(val >= minval);
	assert(val <= maxval);
	return val;
}

bool testDistances(const Geometry<float> &g, const Matrix<char, 3u> &m,
		   const double voxelSize, const int pad)
{
	// D3Q15, I think
	std::vector<iPoint3D> directions;
	directions.push_back(iPoint3D( 0, 0, 0));

	directions.push_back(iPoint3D( 1, 0, 0));
	directions.push_back(iPoint3D( 0, 1, 0));
	directions.push_back(iPoint3D( 0, 0, 1));

	directions.push_back(iPoint3D(-1, 0, 0));
	directions.push_back(iPoint3D( 0,-1, 0));
	directions.push_back(iPoint3D( 0, 0,-1));
/*
	directions.push_back(iPoint3D( 1, 1, 1));
	directions.push_back(iPoint3D( 1, 1,-1));
	directions.push_back(iPoint3D( 1,-1, 1));
	directions.push_back(iPoint3D(-1, 1, 1));
	directions.push_back(iPoint3D(-1,-1, 1));
	directions.push_back(iPoint3D(-1, 1,-1));
	directions.push_back(iPoint3D( 1,-1,-1));
	directions.push_back(iPoint3D(-1,-1,-1));
*/

	std::map<iPoint3D, std::vector<std::pair<int, double> > > dists;
	assert(distances(g, m, voxelSize, directions, dists, pad));
	assert(dists.size() > 0);

	typedef std::map<iPoint3D, std::vector<std::pair<int, double> > >::const_iterator map_const_iterator;
	for (map_const_iterator mi = dists.begin(); mi != dists.end(); ++mi)
	{
//		std::cout << mi->first << std::endl;
		typedef std::vector<std::pair<int, double> >::const_iterator vec_const_iterator;
		for (vec_const_iterator vi = mi->second.begin(); vi != mi->second.end(); ++vi)
		{
			//std::cout << "\t" << vi->first << " " << directions[vi->first].to_string() << " " << vi->second << std::endl;
			assert(vi->first > 0); // should skip (0, 0, 0) direction
//			std::cout << (vi->second - 0.5) << std::endl;
			assert(std::abs(vi->second - 0.5) < 1.0e-15);
		}
	}

	return true;
}

bool testDistances(const Geometry<float> &g, const DTree<char, 3u> &octree,
		   const double voxelSize)
{
	// D3Q15, I think
	std::vector<iPoint3D> directions;
	directions.push_back(iPoint3D( 0, 0, 0));

	directions.push_back(iPoint3D( 1, 0, 0));
	directions.push_back(iPoint3D( 0, 1, 0));
	directions.push_back(iPoint3D( 0, 0, 1));

	directions.push_back(iPoint3D(-1, 0, 0));
	directions.push_back(iPoint3D( 0,-1, 0));
	directions.push_back(iPoint3D( 0, 0,-1));
/*
	directions.push_back(iPoint3D( 1, 1, 1));
	directions.push_back(iPoint3D( 1, 1,-1));
	directions.push_back(iPoint3D( 1,-1, 1));
	directions.push_back(iPoint3D(-1, 1, 1));
	directions.push_back(iPoint3D(-1,-1, 1));
	directions.push_back(iPoint3D(-1, 1,-1));
	directions.push_back(iPoint3D( 1,-1,-1));
	directions.push_back(iPoint3D(-1,-1,-1));
*/

	std::map<std::size_t, std::vector<std::pair<int, double> > > dists;
	assert(distances(g, octree, voxelSize, directions, dists));
	assert(dists.size() > 0);

	typedef std::map<std::size_t, std::vector<std::pair<int, double> > >::const_iterator map_const_iterator;
	for (map_const_iterator mi = dists.begin(); mi != dists.end(); ++mi)
	{
//		std::cout << mi->first << std::endl;
		typedef std::vector<std::pair<int, double> >::const_iterator vec_const_iterator;
		for (vec_const_iterator vi = mi->second.begin(); vi != mi->second.end(); ++vi)
		{
			//std::cout << "\t" << vi->first << " " << directions[vi->first].to_string() << " " << vi->second << std::endl;
			assert(vi->first > 0); // should skip (0, 0, 0) direction
//			std::cout << (vi->second - 0.5) << std::endl;
			assert(std::abs(vi->second - 0.5) < 1.0e-15);
		}
	}

	return true;
}
/*
namespace detail
{



template <typename T, std::size_t D>
void assign(const typename cvmlcpp::DTree<T, D>::DNode &n, cvmlcpp::Matrix<T, D> &m,
	    const cvmlcpp::StaticVector<std::size_t, D> &offset)
{
	const std::size_t N = m.extent(X); // all extents are equal
	const std::size_t L = N / (1ul << n.depth());

	if (n.isLeaf())
		assignValue(m, offset, L);
	else
	{

	}
}

} // end namespace detail

template <typename T, std::size_t D>
cvmlcpp::Matrix<T, D> expand(cvmlcpp::DTree<T, D> &d)
{
	std::array<std::size_t, D> dims;
	const std::size_t N = 1ul<<d.max_depth();
	dims.assign(N);
	cvmlcpp::Matrix<T, D> m(dims.begin());

}
*/
bool testSameOutput(const Geometry<float> &g, const unsigned maxLevel)
{
	bool ok = true;
	double geometrySize = 0;
	for (int i = 0; i < 3; ++i)
	{
		const double d = g.max(i) - g.min(i);
		geometrySize = std::max(d, geometrySize);
	}
	const double voxelSize = geometrySize / double(1ul<<maxLevel);

	//typedef char T;
	typedef short int T; // to put things on screen

	Matrix<T, 3u> m;
	DTree<T, 3u> t(0);
	assert(voxelize(g, m, voxelSize));
	assert(voxelize(g, t, voxelSize));
//std::cout << "DTree depth: " << t.max_depth() << " / " << maxLevel << std::endl;
//std::cout << "Extents: " << m.extent(X) << " " << m.extent(Y) << " " << m.extent(Z) << std::endl;
/*
std::ofstream fm("voxel_ascii_file.dat");
std::ofstream ft("octree_ascii_file.dat");
std::ofstream e("error_ascii_file.dat");
*/
	for (unsigned x = 0; x < m.extent(X); ++x)
	for (unsigned y = 0; y < m.extent(Y); ++y)
	for (unsigned z = 0; z < m.extent(Z); ++z)
	{
		DTree<T, 3u>::DNode n = t.root();
		while (!n.isLeaf())
		{
			unsigned index = 0;
			const unsigned currentLevel = n.depth();
			const std::size_t mask = 1ul << (maxLevel-currentLevel-1);

			index += (x & mask) ? 1 : 0;
			index += (y & mask) ? 2 : 0;
			index += (z & mask) ? 4 : 0;
//std::cout << x << " " << y << " " << z << " mask: " << mask << ": " << " index: " << index << " matrix " << int(m[x][y][z]) << std::endl;
			n = n[index];
			assert(n.depth() <= maxLevel);
		}
/*		fm << int(m[x][y][z]) << " ";
		ft << int(n()) << " ";
		e  << int(m[x][y][z] != n()) << " ";
*/		if (n() != m[x][y][z])
		{
			std::cout << x << " " << y << " " << z << "(" << (x*voxelSize)
				<< ", " << (y*voxelSize) <<", " << (z*voxelSize) <<"): Matrix "
				<< int(m[x][y][z]) << " vs " << int(n()) << " at node " << n.id()
				<< " depth: " << n.depth() << " / " << n.max_depth() << " specified: "
				<< maxLevel << std::endl;
			ok = false;
		}
		assert(n() == m[x][y][z]);
	}
/*	fm.close();
	ft.close();
	e.close();
*/

	return ok;
}


int main(int argc, char **argv)
{
	Geometry<float> g;

	if (argc >= 2)
		assert(readSTL(g, argv[1]));
	else
		assert(readSTL(g, "cube.stl"));

	double voxelSize;

	if (argc == 3)
		voxelSize = std::atof(argv[2]);
	else
		voxelSize = 0.1;

	for (int i = 1; i <= 9; ++i)
		assert(testSameOutput(g, i));

	Matrix<char, 3u> m, m2, m3;
//  	boost::multi_array<char, 3u> m, m2;
//	std::cout << (  fPoint3D(g.max(X), g.max(Y), g.max(Z)) -
//			fPoint3D(g.min(X), g.min(Y), g.min(Z)) ) << std::endl;
	assert(voxelize(g, m, voxelSize, 1));
	if (argc < 2) // only for cube
		assert(testDistances(g, m, voxelSize, 1));

	DTree<char, 3u> octree(0);
	assert(voxelize(g, octree, 1.0/32.0));
	if (argc < 2) // only for cube
		assert(testDistances(g, octree, 1.0/32.0));

	DTree<int, 3u> d(0);
	assert(voxelize(g, d, 1.0/32.0));
//	std::cout << d << std::endl;
	unsigned elems = 0u;
	if (argc < 2) // only for cube
	{
		for (unsigned x = 0u; x < m.extents()[X]; ++x)
		for (unsigned y = 0u; y < m.extents()[Y]; ++y)
		for (unsigned z = 0u; z < m.extents()[Z]; ++z)

	// 	for (unsigned x = 0u; x < 12; ++x)
	// 	for (unsigned y = 0u; y < 12; ++y)
	// 	for (unsigned z = 0u; z < 12; ++z)

		{
			if ( (x == 0) || (y == 0) || (z == 0) ||
			     (x > 10) || (y > 10) || (z > 10) )
			{
				if (m[x][y][z] != 0)
				std::cout << "(" << x << ", " << y << ", " << z << ") "
					<< int(m[x][y][z]) << std::endl;
				assert(m[x][y][z] == 0);
			}
			else
			{
				if (m[x][y][z] != 1)
				std::cout << "(" << x << ", " << y << ", " << z << ") "
					<< int(m[x][y][z]) << std::endl;
				assert(m[x][y][z] == 1);
			}

			if (m[x][y][z] == 1)
				++elems;
		}
	// 	std::cout << "Elements " << elems << std::endl;
		assert(elems == 1000u);

		assert(voxelize(g, m, 0.2, 0));
		elems = 0u;
		for (unsigned x = 0u; x < m.extents()[X]; ++x)
		for (unsigned y = 0u; y < m.extents()[Y]; ++y)
		for (unsigned z = 0u; z < m.extents()[Z]; ++z)
		{
			if ( (x >= 5) || (y >= 5) || (z >= 5) )
			{
				if (m[x][y][z] != 0)
				std::cout << "0.2 (" << x << ", " << y << ", " << z << ") "
					<< int(m[x][y][z]) << std::endl;
	// 			assert(m[x][y][z] == 0);
			}
			else
			{
				if (m[x][y][z] != 1)
				std::cout << "0.2 (" << x << ", " << y << ", " << z << ") "
					<< int(m[x][y][z]) << std::endl;
	// 			assert(m[x][y][z] == 1);
			}

			if (m[x][y][z] == 1)
				++elems;
		}
	// 	std::cout << "Elements: " << elems << std::endl;
		assert(elems == 125u);

		// Test octree
		assert(voxelize(g, octree, voxelSize));
		const std::size_t N = 1ul << octree.max_depth();
//std::cout << "octree dimensionality " << N << std::endl;
		elems = 0u;
	//std::ofstream ft("octree_ascii_file.dat");
		for (unsigned x = 0u; x < N; ++x)
		for (unsigned y = 0u; y < N; ++y)
		for (unsigned z = 0u; z < N; ++z)
		{
			const int value = octree.getValue(x, y, z);
			/*if ( (x >= 10) || (y >= 10) || (z >= 10) )
			{
				if (value != 0)
					std::cout << "octree error (" << x << ", " << y << ", " << z << ") "
						<< value << std::endl;
				assert(value == 0);
			}
			else
			{
				if (value != 1)
					std::cout << "octree error (" << x << ", " << y << ", " << z << ") "
						<< value << std::endl;
				assert(value == 1);
			}*/
			//ft << value << " ";
			if (value)
				++elems;
		}
		//ft.close();
		//std::cout << "Elements: " << elems << " Ratio: " << (float(elems)/N/N/N)
			//<< " expected: " << (125./8./8./8.) << std::endl;
		assert(std::abs(float(elems)/N/N/N - 125./8./8./8.) < 0.00001);
	}

	assert(writeArray(m, "/tmp/m2.dat"));
	assert(readArray(m2, "/tmp/m2.dat"));
	for (unsigned i = 0u; i < 3u; ++i)
		assert(m.extent(i) == m2.extent(i));

	for (Matrix<char, 3u>::const_iterator i = m.begin(), j = m2.begin();
	     i != m.end(); ++i, ++j)
		assert(*i == *j);

	assert(writeArray(m,  "/tmp/m3.dat", false));
	assert(readArray (m3, "/tmp/m3.dat"));
	for (unsigned i = 0u; i < 3u; ++i)
		assert(m.extent(i) == m3.extent(i));

	for (Matrix<char, 3u>::const_iterator i = m.begin(), j = m3.begin();
	     i != m.end(); ++i, ++j)
		assert(*i == *j);

	/*
	 * Subvoxels
	 */
	Matrix<float, 3u> fract;
	assert(fractionVoxelize(g, fract, voxelSize, 5, 1));
	double weight = 0.0;
	if (argc < 2) // only for cube
	{
		for (unsigned x = 0u; x < 12; ++x)
		for (unsigned y = 0u; y < 12; ++y)
		for (unsigned z = 0u; z < 12; ++z)
			weight += fract[x][y][z];
		if (!(std::abs(weight - 1000.0) < 0.01))
			std::cout << "Weight: " << weight << " != 1000" << std::endl;
		assert(std::abs(weight - 1000.0) < 0.01);
	}

	// From the Fraunhofer institute
	for (int t = 0; t < 100; ++t)
	{
	        srand48(t);
		g.clear();

		const double lim = double(t) / 10.;
		unsigned fnumber = rnd(0., 64.);
		for (unsigned i = 0; i < fnumber; ++i)
		{
			const double p0v0 = rnd(-lim, 1.+lim);
			const double p0v1 = rnd(-lim, 1.+lim);
			const double p0v2 = rnd(-lim, 1.+lim);
			const double p1v0 = rnd(-lim, 1.+lim);
			const double p1v1 = rnd(-lim, 1.+lim);
			const double p1v2 = rnd(-lim, 1.+lim);
			const double p2v0 = rnd(-lim, 1.+lim);
			const double p2v1 = rnd(-lim, 1.+lim);
			const double p2v2 = rnd(-lim, 1.+lim);

//			std::cout << "Facet(" << i << ") = ["
// 			<< "(" << p0v0 << "," << p0v1 << "," << p0v2 << "), "
// 			<< "(" << p1v0 << "," << p1v1 << "," << p1v2 << "), "
// 			<< "(" << p2v0 << "," << p2v1 << "," << p2v2 << ")]"
// 			<< std::endl;

			g.addFacet(g.addPoint(p0v0, p0v1, p0v2),
				   g.addPoint(p1v0, p1v1, p1v2),
				   g.addPoint(p2v0, p2v1, p2v2));
		}

		voxelize(g, m, rnd(0.05, (1.+t)/10.));
	}

	return 0;
}
