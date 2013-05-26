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
#include <vector>

#include <cvmlcpp/base/Matrix>
#include <cvmlcpp/signal/Processing>

#include <boost/multi_array.hpp>

using namespace cvmlcpp;

template <typename T>
bool testProcessing()
{
	unsigned dims[] = {3, 4, 5};
	Matrix<T, 3> a(dims, 1);
	Matrix<T, 3> b(dims, 2);
	std::vector<Matrix<T, 3> > bv;
	bv.push_back(a);
	bv.push_back(b);

	Matrix<T, 3> r;
	std::vector<Matrix<T, 3> > rv;

	convolute(a, b, r);
	correlate(a, b, r);
	autocorrelate(a, r);
	autocorrelate(bv, rv);

	boost::multi_array<T, 3> A(boost::extents[3][4][5]);
	boost::multi_array<T, 3> B(boost::extents[3][4][5]);

	boost::multi_array<T, 3> R;

	convolute(A, B, R);
	correlate(A, B, R);
	autocorrelate(A, R);
	return true;
}

int main()
{
	testProcessing<float>();
	testProcessing<double>();
#ifndef USE_CUFFT
	testProcessing<long double>();
#endif

	return 0;
}
