/***************************************************************************
 *   Copyright (C) 2007 by BEEKHOF, Fokko                                  *
 *   fpbeekhof@gmail.com                                                   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; version 3 of the License.               *
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

#include <string>
#include <map>
#include <cassert>
#include <cstdio>

#include <boost/lexical_cast.hpp>

namespace cvmlcpp
{

namespace detail
{

template <class T>
std::map<std::string, T> &storage()
{
	static std::map<std::string, T> s;

	return s;
}

} // end namespace detail

template <class T>
bool hasKey(std::string key)
{
	if (nameSpace() != "")
		key = nameSpace() + std::string("::") + key;

	return (detail::storage<T>().find(key) != detail::storage<T>().end());
}

template <class T>
bool hasKey(std::string key, std::size_t index)
{
	if (nameSpace() != "")
		key = nameSpace() + std::string("::") + key;

	key += "[";
	key += boost::lexical_cast<std::string>(index);
	key += "]";

	return (detail::storage<T>().find(key) != detail::storage<T>().end());
}

template <class T>
void store(std::string key, const T value)
{
	if (nameSpace() != "")
		key = nameSpace() + std::string("::") + key;

	detail::storage<T>()[key] = value;
}

template <class T>
void store(std::string key, const T value, std::size_t index)
{
	if (nameSpace() != "")
		key = nameSpace() + std::string("::") + key;

	key += "[";
	key += boost::lexical_cast<std::string>(index);
	key += "]";

	store<T>(key, value);
}

template <class T>
T &retrieve(std::string key)
{
	if (nameSpace() != "")
		key = nameSpace() + std::string("::") + key;

	assert(detail::storage<T>().find(key) != detail::storage<T>().end());

	return detail::storage<T>()[key];
}

template <class T>
T &retrieve(std::string key, std::size_t index)
{
	key += "[";
	key += boost::lexical_cast<std::string>(index);
	key += "]";

	return retrieve<T>(key);
}

} // end namespace
