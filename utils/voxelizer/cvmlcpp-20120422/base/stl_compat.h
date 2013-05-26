/***************************************************************************
 *   Copyright (C) 2007 by BEEKHOF, Fokko                                  *
 *   fpbeekhof@gmail.com                                                   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2u of the License, or     *
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

#ifndef CVMLCPP_STL_COMPAT_H
#define CVMLCPP_STL_COMPAT_H 1

#ifdef __GNUC__

	#define _HAVE_TR1 1

	#if (__GNUC__ >= 4)
		#if (__GNUC_MINOR__ >= 2)
			#define _HAVE_TR1_CCTYPE  1
			#define _HAVE_TR1_CMATH   1
			#define _HAVE_TR1_CSTDINT 1
		#endif
	#endif

#endif

#endif // STL_COMPAT_H
