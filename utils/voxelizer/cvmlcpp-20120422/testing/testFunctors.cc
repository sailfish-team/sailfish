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

#include <vector>
#include <algorithm>

#include <cvmlcpp/base/Functors>

int main()
{
	using namespace cvmlcpp;
	std::vector<std::vector<int> > a(10);

	a[3].push_back(10);
	a[3].push_back(5);

	ContainerSorter<std::vector<int> > srt;
	std::for_each(a.begin(), a.end(), srt);

	float b = 1.f, c = 2.f;
	roughly_equal_to<float> eq;
	assert(!eq(b, c));

	return 0;
}
