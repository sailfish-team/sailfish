/***************************************************************************
 *   Copyright (C) 2009 by BEEKHOF, Fokko                                  *
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
#include <iostream>
#include <cvmlcpp/math/Splines>
#include <cvmlcpp/math/Euclid>

int main()
{
	using namespace cvmlcpp;
	
	double data [] = {0, 1, 2, 3};
	NaturalCubicSpline<double, 1> spline(data, data+4);

//	for (double d = 0.0; d <= spline.size()+0.05; d += 0.1)
		/*std::cout << spline(d) << " "*/;
//	std::cout << std::endl;
	for (int i = 0; i < 4; ++i)
		assert(spline(i) == i);

	std::vector<dPoint2D> data2;
	for (int i = 0; i < 4; ++i)
	{
		data2.push_back( dPoint2D(i, 2*i) );
		//~ assert(data2[i][0] == i);
		//~ assert(data2[i][1] == i);
	}
	NaturalCubicSpline<dPoint2D, 2> spline2(data2.begin(), data2.end());
	
	//~ for (double d = 0.0; d <= spline.size()+0.05; d += 0.1)
		//~ std::cout << spline2(d) << " ";
	//~ std::cout << std::endl;
	for (int i = 0; i < 4; ++i)
	{
		assert(spline2(i).x() == i);
		assert(spline2(i).y() == 2*i);
	}

	return 0;
}

