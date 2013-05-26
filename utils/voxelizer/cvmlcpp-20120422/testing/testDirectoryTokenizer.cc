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

#include <iostream>
#include <cvmlcpp/base/DirectoryTokenizer>

int main(int argc, char **argv)
{
/*	if (argc != 2)
	{
		std::cout << "Usage: " << argv[0] << " <directory>" <<std::endl;
		return 0;
	}

	DirectoryTokenizer dt(argv[1]);
*/	cvmlcpp::DirectoryTokenizer dt(".");
	assert(dt.ok());
	std::string fn;

	unsigned entries = 0u;
	while(dt.next(fn))
	{
		++entries;
// 		std::cout << fn << std::endl;
	}
	assert(entries > 0);

	return 0;
}
