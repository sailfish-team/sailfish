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

#include <cassert>
#include <complex>
#include <numeric>

#include <cvmlcpp/array/BlitzArray>
#include <cvmlcpp/signal/Fourier>
#include <cvmlcpp/signal/Processing>

#include <cvmlcpp/volume/Geometry>
#include <cvmlcpp/volume/Voxelizer>
#include <cvmlcpp/volume/VolumeIO>

using namespace cvmlcpp;

int main()
{
	BlitzArray<int, 3> p(3, 4, 5);
	BlitzArray<int, 2> q = p[2];

	/*
	 * 3D Fourier Transform
	 */
	// Have a blitz++ array
	blitz::Array<int, 3> a(3, 4, 5);
	std::fill(a.begin(), a.end(), 7);

	// Create a wrapper around "a" for cvmlcpp-compatibility
	BlitzArray<int, 3> ba(a);
	assert( (3*4*5*7) == std::accumulate(ba.begin(), ba.end(), 0));

	// Create a wrapper for the output
//	blitz::Array<std::complex<double>, 3> a_fft(3, 4, 5);
//	BlitzArray<std::complex<double>, 3> ba_fft(a_fft);
	BlitzArray<std::complex<double>, 3> ba_fft;

	// Compute 3D-Fourier tranform with "fftw"
	doDFT(ba, ba_fft);

	/*
	 * Correlation
	 */
	blitz::Array<int, 3> b(3, 4, 5);
	std::fill(b.begin(), b.end(), 7);

	BlitzArray<int, 3> bb(b);
	BlitzArray<double, 3> bc;

 	correlate(ba, bb, bc);

	blitz::Array<double, 3> c;
	c = bc;

	/*
	 * Voxelize a 3D geometry
	 */
	BlitzArray<char, 3> m;
	Geometry<float> g;
	assert(readSTL(g, "cube.stl"));
	assert(voxelize(g, m, 0.1, 1));

	unsigned elems = 0u;
	for (BlitzArray<char, 3u>::const_iterator i = m.begin(); i != m.end(); ++i)
	{
		assert(*i == 0 || *i == 1);
		elems += *i;
	}
	assert(elems == 1000u);

	/*
	 * I/O
	 */
	// Write to disk
	assert(writeArray(m, "/tmp/m2.dat"));

	// Read back
	BlitzArray<char, 3> m2;
	assert(readArray(m2, "/tmp/m2.dat"));

	for (BlitzArray<char, 3u>::const_iterator i = m.begin(), j = m2.begin();
	     i != m.end(); ++i, ++j)
		assert(*i == *j);

	return 0;
}
