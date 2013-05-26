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

#include <iostream>

#include <cvmlcpp/math/Vectors>
#include <cvmlcpp/math/Euclid>

using namespace cvmlcpp;

int main()
{
	StaticVector<int, 3u> v1 = 3;
	DynamicVector<int> d1(3, 3);

	StaticVector<int, 3u> v2 = v1 + 3;
	v2 += d1;
	StaticVector<int, 3u> v3 = v1 + 3 + d1;
std::cout << "error: no match for ‘operator==’ in ‘9 == v2’" << std::endl;
//	assert(9 == v2);

	StaticVector<int, 3u> v4 = min(v1, v2);
	StaticVector<int, 3u> v5 = max(v1, v2);
	for (int i = 0; i < v1.size(); ++i)
	{
		assert(v1[i] <= v2[i]);
		assert(v4[i] == v1[i]);
		assert(v5[i] == v2[i]);
	}

	const StaticVector<bool, 80>::register_type raw [] =
		{0x824a29d3, 0xb79088d9, 0x82f27ae1};
	StaticVector<bool, 80> bv1(raw, raw + 3u);

	StaticVector<bool, 80> bv2(bv1.to_string());
	assert(bv1.to_string() == bv2.to_string());

	return 0;
}
