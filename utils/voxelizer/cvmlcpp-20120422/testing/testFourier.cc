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

#include <complex>
#include <iostream>

#include <cvmlcpp/base/Enums>
#include <cvmlcpp/math/Math>
#include <cvmlcpp/base/Matrix>
#include <cvmlcpp/signal/Fourier>
#include <omptl/omptl_algorithm>

template<typename float_type>
void testFourier()
{
	using namespace cvmlcpp;

//	const std::size_t dims [] = { 1024u, 4096u };
	const std::size_t dims [] = { 64u, 128u };
// 	const std::size_t dims [] = { 4u, 8u };

	Matrix<float_type, 2u> s(dims);

	for (unsigned i = 0u; i < s.size(); ++i)
		*(s.begin() + i) = std::sin(i*PI/4.0);

// 	for (std::size_t x = 0; x < dims[X]; ++x)
// 	{
// 		for (std::size_t y = 0; y < dims[Y]; ++y)
// 			std::cout << s[x][y] << "   ";
// 		std::cout << std::endl;
// 	}
// 	std::cout << std::endl;

	Matrix<std::complex<float_type>, 2u> f;
	assert(doDFT(s, f, 2u));
	assert(f.extent(X) ==  dims[X]);
	assert(f.extent(Y) == (dims[Y]/2u)+1u);

// 	for (std::size_t x = 0; x < dims[X]; ++x)
// 	{
// 		for (std::size_t y = 0; y <= dims[Y]/2; ++y)
// 			std::cout << f[x][y] << "   ";
// 		std::cout << std::endl;
// 	}
// 	std::cout << std::endl;

	Matrix<float_type, 2u> s2;
	assert(doDFT(f, s2, 2u));
	assert(s2.extent(X) == dims[X]);
	assert(s2.extent(Y) == dims[Y]);

// 	for (std::size_t x = 0; x < dims[X]; ++x)
// 	{
// 		for (std::size_t y = 0; y < dims[Y]; ++y)
// 			std::cout << s2[x][y] << "   ";
// 		std::cout << std::endl;
// 	}
// 	std::cout << std::endl;

	for (unsigned i = 0u; i < s2.size(); ++i)
	{
		if (!( std::abs(*(s2.begin() + i) - std::sin(i*PI/4.0)) <
			0.00001))
			std::cout << i << ": " << *(s2.begin() + i) << " "
				<< std::sin(i*PI/4.0) << std::endl;
		assert( std::abs(*(s2.begin() + i) - std::sin(i*PI/4.0)) <
			0.00001);
	}

	assert(doDFT(s, f, 2u));
	assert(doDFT(f, true, 2u));
	assert(doDFT(f, false, 2u));

	assert(doDFT(f, s2, 2u));
	for (unsigned i = 0u; i < s2.size(); ++i)
	{
		if (!( std::abs(*(s2.begin() + i) - std::sin(i*PI/4.0)) <
			0.00001))
			std::cout << i << ": " << *(s2.begin() + i) << " "
				<< std::sin(i*PI/4.0) << std::endl;
		assert( std::abs(*(s2.begin() + i) - std::sin(i*PI/4.0)) <
			0.00001);
	}

	Matrix<int, 2u> si(dims);
	Matrix<std::complex<float_type>, 2u> fi(dims);
	Matrix<int, 2u> si2(dims);

	for (unsigned i = 0u; i < si.size(); ++i)
		*(si.begin() + i) = i & 1;

	assert(doDFT(si, fi, 2u));
	assert(doDFT(fi, si2, 2u));

	for (unsigned i = 0u; i < si2.size(); ++i)
	{
		if (*(si2.begin() + i) != (int)(i & 1))
			std::cout << i << ": " << *(si2.begin() + i) << " "
				<< (i & 1) << std::endl;
		assert( *(si2.begin() + i) == (int)(i & 1) );
	}

	Matrix<int, 2u> si3 = si2.clone();
	fftshift(si3);
	fftshift(si3);
	for (unsigned i = 0u; i < si2.size(); ++i)
		assert( *(si2.begin() + i) == *(si3.begin() + i) );

	const unsigned mask_size_1 = 9, mask_size[] = {mask_size_1};
	Matrix<float_type, 1u> mask(mask_size, 1./mask_size_1);
	Matrix<std::complex<float_type>, 1u> mask_dft(mask_size);
	assert(doDFT(mask, mask_dft));

	Matrix<float_type, 1u> fi3 = mask.clone();
	fftshift(fi3);
	fftshift(fi3);
	for (unsigned i = 0u; i < mask.size(); ++i)
		assert( mask[i] == fi3[i] );
}

int main()
{
	testFourier<float>();
	testFourier<double>();
#ifndef USE_CUFFT
	testFourier<long double>();
#endif

	const std::size_t dims [] = { 3u, 3u };
	const int data [] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	cvmlcpp::Matrix<int, 2u> m(dims);
	std::copy(data, data+m.size(), m.begin());
/*	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
			std::cout << m[i][j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
*/
	cvmlcpp::fftshift(m);
/*
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
			std::cout << m[i][j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
*/	
	cvmlcpp::Matrix<int, 1u> m1(dims);
	std::copy(data, data+m1.size(), m1.begin());
/*	for (int i = 0; i < 3; ++i)
		std::cout << m1[i] << " ";
	std::cout << std::endl;
*/	
	cvmlcpp::fftshift(m1);
/*
	for (int i = 0; i < 3; ++i)
		std::cout << m1[i] << " ";
	std::cout << std::endl;
*/	
	return 0;
}
