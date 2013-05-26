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
#include <cassert>

#include <cvmlcpp/base/Enums>
#include <cvmlcpp/base/IDGenerators>
#include <cvmlcpp/base/Matrix>

using namespace cvmlcpp;

int main()
{
	const unsigned dims [] = {4, 5, 6};
	const unsigned dimsEq [] = {4, 4, 4};

	// Full
	Matrix<bool, 2> mbool(dims);
	mbool[2][2] = true; // Test assignment

	// Full
	Matrix<int, 3> m1(dims);
	assert(m1.size() == std::accumulate(dims, dims+3, 1u,
					std::multiplies<unsigned>()));

	std::fill(m1.begin(), m1.end(), 4);
	assert(std::accumulate(m1.begin(),m1.end(),0) == (int)(4u*m1.size()));


	m1[1][3][2] = 3;
	assert(m1[1][3][1] == 4);
	assert(m1[1][3][2] == 3);
	assert(m1[1][3][3] == 4);

	m1.resize(dimsEq);
	assert(m1.size() == 64u);
	assert(std::accumulate(m1.begin(),m1.end(),0) == (int)(4u*m1.size()-1));


	const unsigned dimsr [] = {2, 3};
	const unsigned dimsc [] = {dimsr[1], dimsr[0]};
	Matrix<int, 2> m2dr(dimsr, -1, false);
	Matrix<int, 2> m2dc(dimsc, -1, true);
// std::cout<< "Row-Major:" << std::endl;
	for (unsigned ctr = 0, r = 0; r < m2dr.extent(0); ++r)
	{
		for (unsigned c = 0; c < m2dr.extent(1); ++c)
		{
			assert( &(*(m2dr.begin()+r*dimsr[1]+c)) == &m2dr[r][c]);
			assert(m2dr[r][c] == -1);
			m2dr[r][c] = ctr++;
// 			std::cout << m2dr[r][c] << " ";
		}
// 		std::cout << std::endl;
	}

// std::cout<< "Col-Major:" << std::endl;
	for (unsigned c = 0; c < m2dc.extent(0); ++c)
	{
		for (unsigned r = 0; r < m2dc.extent(1); ++r)
		{
// std::cout << std::endl << "(" << c << ", " << r << ") IDX: "
// 	<< (c*dimsc[1]+r) << std::endl;

			assert( &(*(m2dc.begin()+c*dimsc[1]+r)) == &m2dc[c][r]);

// 			std::cout << "(" << c << ", " << r << ") " <<
// 				m2dc[c][r] << " --> " << m2dr[r][c] <<std::endl;
			assert(m2dc[c][r] == -1);
			m2dc[c][r] = m2dr[r][c];
// 			std::cout << m2dc[c][r] << std::endl;
		}
// 		std::cout << "--"  << std::endl;
	}

	for (unsigned r = 0; r < m2dr.extent(0); ++r)
	for (unsigned c = 0; c < m2dr.extent(1); ++c)
	{//offset = row*NUMCOLS + column

		assert(m2dr[r][c] == int(r*m2dr.extent(1) + c));
		assert(m2dc[c][r] == m2dr[r][c]);
	}

// for (Matrix<int, 2>::const_iterator i = m2dr.begin(); i != m2dr.end(); ++i)
// 	std::cout << *i << " "; std::cout << std::endl;
// for (Matrix<int, 2>::const_iterator i = m2dc.begin(); i != m2dc.end(); ++i)
// 	std::cout << *i << " "; std::cout << std::endl;

/*
	const unsigned dimsL [] = {1024, 1024, 64};
	Matrix<int, 3> m(dims);
	m.resize(dimsL);
	std::fill(m.begin(), m.end(), 2);
	std::cout << std::accumulate(m.begin(), m.end(), 0) << std::endl;
*/
	// Compressed
	CompressedMatrix<int, 3> m2(dims);
	assert(m2.size() == std::accumulate(dims, dims+3, 1u,
					std::multiplies<unsigned>()));
	std::fill(m2.begin(), m2.end(), 4);

// std::cout << "END FILL ------------------ " << std::endl;
// 	std::for_each(m2.begin(), m2.end(), functors::Printer<int>());

	assert(std::accumulate(m2.begin(), m2.end(), 0) == (int)(4u*m2.size()));

	m2.resize(dimsEq);
	assert(m2.size() == std::accumulate(dimsEq, dimsEq+3, 1u,
					std::multiplies<unsigned>()));
	std::fill(m2.begin(), m2.end(), 4);

	m2[1][3][2] = 3;
	assert(m2[1][3][1] == 4);
	assert(m2[1][3][2] == 3);
	assert(m2[1][3][3] == 4);
	assert(std::accumulate(m2.begin(),m2.end(),0) == (int)(4*m2.size()-1));

	// Dynamic
	DynamicMatrix<float> matrix1(dims, dims+3, 1.0f);

	matrix1[2][2][2] = 9.0f;
	assert(matrix1[2][2][2] == 9.0f);
	assert(matrix1[2][2][1] == 1.0f);
	assert(std::accumulate(matrix1.begin(), matrix1.end(), 0.0f) ==
		(float)matrix1.size() + 8.0f);

	// Methode 2
	std::vector<unsigned> vdims;
	vdims.push_back(7);
	vdims.push_back(9);
	DynamicMatrix<int> matrix2(vdims.begin(), vdims.end(), 1);
	matrix2[6][5] = -1;
	assert(std::accumulate(matrix2.begin(), matrix2.end(), 0) ==
		(int)matrix2.size() - 2);

	// Symmetric Matrix
	const std::size_t K = 16;
	SymmetricMatrix<int> sm(K, -1);
	for (std::size_t i = 0, k = 0; i < K; ++i)
	for (std::size_t j = i; j < K; ++j, ++k)
		sm(i, j) = k;

	for (std::size_t i = 0, k = 0; i < K; ++i)
	for (std::size_t j = i; j < K; ++j, ++k)
	{
		assert(sm(i, j) == int(k));
		assert(sm(j, i) == int(k));
	}

	return 0;
}
