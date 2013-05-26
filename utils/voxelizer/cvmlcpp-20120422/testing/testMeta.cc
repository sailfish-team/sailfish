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

// #ifndef USE_BLITZ
// #define USE_BLITZ
// #endif

#include <vector>

#include <cvmlcpp/base/Meta>

using namespace cvmlcpp;

int main()
{
	array_traits<Matrix, int, 2u, std::vector<int> >::value_type	a = 3;
	array_traits<boost::multi_array, int, 2u, std::allocator<int> >::
		value_type b = 3;
// 	array_traits<blitz::Array, int, 2u>::value_type		c = 3;

	assert(a == b);

	char buf [50]; // Should convert to double, and thus compile without warning
	sprintf(buf, "%f", promote_trait1<std::complex<int> >::value_type(0));

	return 0;
}

