/***************************************************************************
 *   Copyright (C) 2008 by BEEKHOF, Fokko                                  *
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

#include <cvmlcpp/base/Holder>
#include <cassert>
#include <string>

#include <iostream>
#include <sstream>

#include <boost/lexical_cast.hpp>

/*
 * The framework consists of 3 elements
 *
 * 1) The set of polymorphic classes
 * 2) A Factory-method implemented as a Functor
 * 3) An operator<<()
 */

// First element: The set of polymorphic classes
struct Furniture
{
	int price;

	virtual ~Furniture() { }

	virtual std::string to_string() const = 0;
};

struct Chair : public Furniture
{
	int legs;
	virtual std::string to_string() const
	{
		return  std::string("<chair> <legs> ") +
			boost::lexical_cast<std::string>(legs) +
			std::string(" </legs> </chair>");
	}
};

struct Table : public Furniture
{
	int surface;
	virtual std::string to_string() const
	{
		return  std::string("<table> <surface> ") +
			boost::lexical_cast<std::string>(surface) +
			std::string(" </surface> </table>");
	}
};

// Second element: A Factory-method implemented as a Functor
class FurnitureFactory
{
	public:
		// The factory should produce an object from the class hierarchy
		// given an input stream. It may throw an exception of type
		// cvmlcpp::ParseError if the input is invalid.
		Furniture * operator()(std::istream& i_stream)
			throw(cvmlcpp::ParseError)
		{
			i_stream >> tag; // read open tag
			if (tag == "<chair>")
				return parseChair(i_stream);
			else if (tag == "<table>")
				return parseChair(i_stream);
			else throw (cvmlcpp::ParseError()); // Unknown tag

			assert(false); // We should never get here
			return (Furniture *)0;
		}

		Chair* parseChair(std::istream& i_stream)
			throw(cvmlcpp::ParseError)
		{
			Chair* chair = new Chair();

			// This is buggy but let's keep the example simple
			try {
				// Read "<legs> value </legs>"
				i_stream >> tag >> chair->legs >> tag;
				i_stream >> tag; // read "</chair>"
			}
			catch (std::exception &e) {
				delete chair;
				throw(cvmlcpp::ParseError());
			}

			return chair;
		}

		Table* parseTable(std::istream& i_stream)
			throw(cvmlcpp::ParseError)
		{
			Table* table = new Table();

			// This is buggy but let's keep the example simple
			try {
				// Read "<surface> value </surface>"
				i_stream >> tag >> table->surface >> tag;
				i_stream >> tag; // read "</table>"
			}
			catch (std::exception &e) {
				delete table;
				throw(cvmlcpp::ParseError());
			}

			return table;
		}

	private:
		std::string tag;
};

// Third element: an output operator
std::ostream& operator<<(std::ostream& o_stream, const Furniture &furniture)
{ return o_stream << furniture.to_string(); }

// Needed for test
bool operator==(const Chair& lhs, const Chair& rhs)
{ return lhs.legs == rhs.legs; }

int main()
{
	const std::string chairxml = "<chair> <legs> 4 </legs> </chair>";
	std::stringstream input(chairxml);

	typedef cvmlcpp::Holder<Furniture, FurnitureFactory> FurnitureHolder;
	FurnitureHolder holder1;
	input >> holder1;

	std::stringstream output;
	output << holder1;
	
	FurnitureHolder holder2;
	output >> holder2;

	assert(holder1.cast<Chair>() == holder2.cast<Chair>());

	return 0;
}
