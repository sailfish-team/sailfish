/***************************************************************************
 *   Copyright (C) 2008 by BEEKHOF, Fokko                                  *
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

#ifndef CVMLCPP_STL_COMPAT_CSTDINT_H
#define CVMLCPP_STL_COMPAT_CSTDINT_H 1

#include <cvmlcpp/base/stl_compat.h>

#ifdef _HAVE_TR1_CSTDINT
    #include <cstdint>
#else
    #include <boost/cstdint.hpp>

    typedef boost::int8_t  int8_t;
    typedef boost::int16_t int16_t;
    typedef boost::int32_t int32_t;
    typedef boost::int64_t int64_t;

    typedef boost::uint8_t  uint8_t;
    typedef boost::uint16_t uint16_t;
    typedef boost::uint32_t uint32_t;
    typedef boost::uint64_t uint64_t;
#endif

#endif
