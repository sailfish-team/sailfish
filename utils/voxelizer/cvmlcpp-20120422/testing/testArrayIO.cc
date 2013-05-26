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
#include <functional>
#include <cassert>
#include <cstdlib>

#include <cvmlcpp/base/Matrix>
#include <cvmlcpp/array/ArrayIO>
#include <cvmlcpp/base/IDGenerators>

using namespace cvmlcpp;

int rndi()
{
	return rand() % 2;
}

double rndd()
{
	return drand48();
}

template <template <typename Tm, std::size_t D, typename A> class Array,
	  typename T, std::size_t N, typename Aux>
void test(const Array<T, N, Aux> &m)
{
	 const std::string tmp = getenv ("TMPDIR") ? getenv ("TMPDIR") : "/tmp";

	assert(writeArray(m, tmp+"/m.dat", false));
	assert(writeArray(m, tmp+"/mc.dat", true));

	Array<T, N, Aux> m_verify;
	assert(readArray(m_verify, tmp+"/m.dat"));
	assert(m == m_verify);
	std::fill(m_verify.begin(), m_verify.end(), T(0));
// 	m_verify.clear();
// 	assert(m_verify != );
	assert(readArray(m_verify, tmp+"/mc.dat"));
	assert(m == m_verify);
}

int main()
{
	const unsigned max_size = 256;
	for (int i = 0; i < 64; ++i)
	{
		unsigned dims [3];
		dims[X] = rand() % max_size;
		dims[Y] = rand() % max_size;
		dims[Z] = rand() % max_size;

		Matrix<int, 3> m3(dims);
		std::generate(m3.begin(), m3.end(), IncGenerator<int>());
		test(m3);
		std::generate(m3.begin(), m3.end(), rand);
		test(m3);
		std::generate(m3.begin(), m3.end(), rndi);
		test(m3);

		// Floating-point
		dims[X] = rand() % max_size;
		dims[Y] = rand() % max_size;
		Matrix<double, 2> m2(dims);
		std::generate(m2.begin(), m2.end(), drand48);
		test(m2);
	}

	return 0;
}
