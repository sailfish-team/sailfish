/***************************************************************************
 *   Copyright (C) 2007,2008 by BEEKHOF, Fokko                             *
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

#include <bitset>
#include <cmath>
#include <map>
#include <utility>
#include <vector>
#include <cassert>

#include <tr1/array>

#include <boost/integer/static_log2.hpp>

namespace cvmlcpp
{

/*
 * Interface of a node for message passing algorithms.
 */
class _Node
{
	public:
		_Node() : _cached(0.0) {}

		void setID(const std::size_t id) { _id = id; }

		virtual ~_Node() {}

		void addLink(const _Node * const _node)
		{
			_links.push_back(_node);
			_received[_node->id()] = double(0);
		}

		void calculate()
		{
			for (std::size_t i = 0; i < _links.size(); ++i)
				_received[_links[i]->id()] =
					  _links[i]->messageTo(this->id());

			this->updateCache();
		}

		std::size_t id() const { return _id; }

		void init(const double value = 0.0)
		{
			for (std::size_t i = 0; i < _links.size(); ++i)
				_received[_links[i]->id()] = double(0);

			this->initCache(value);
		}

		virtual double messageTo(const std::size_t dest) const = 0;

		virtual bool bit() const { return false; }

#ifndef NDEBUG
		double value() const { return cache(); }

		void print() const
		{
			std::cout << "ID: " << _id << " Links:";
			for (std::size_t i = 0; i < _links.size(); ++i)
				std::cout << " " << _links[i]->id();
			std::cout << " Value: " << value() << std::endl;
		}
#endif

	protected:
		typedef std::map<std::size_t, double>::const_iterator
			const_iterator;

		virtual void updateCache() = 0;

		virtual void initCache(const double value = 0.0)
		{ cache() = value; }

		double &cache() { return _cached; }

		const double &cache() const { return _cached; }

		const_iterator begin() const { return _received.begin(); }
		const_iterator end  () const { return _received.end  (); }

		double receivedFrom(const std::size_t source) const
		{ return _received[source]; }

	private:
		std::size_t _id;
		double _cached;
		std::vector<const _Node *> _links;
		mutable std::map<std::size_t, double> _received;
};

/*
 * Variable Node implementation for LogLikelihood quantities
 */
class _LLVariableNode : public _Node
{
	public:
		virtual double messageTo(const std::size_t dest) const
		{
			return this->cache() - this->receivedFrom(dest);
		}

		virtual bool bit() const { return this->cache() < double(0); }

	protected:
		// p = Pr[var == 1]
		virtual void initCache(const double p)
		{
			assert ( (p >= 0.0) && (p <= 1.0) );

			this->cache() = std::log( (double(1)-p) / p );
			assert( (this->cache()<=0.0) || (this->cache()>=0.0) );
		}

		virtual void updateCache()
		{
			double sum = 0.0;
			for (_Node::const_iterator it=this->begin();
			     it != this->end(); ++it)
			{
				assert((it->second>=0.0) || (it->second<=0.0));

				sum += it->second;
				sum = std::min(sum,
					 std::numeric_limits<double>::max());
				sum = std::max(sum,
					-std::numeric_limits<double>::max());
				assert( (sum >= 0.0) || (sum <= 0.0) );
			}

			assert( (sum >= 0.0) || (sum <= 0.0) );

			this->cache() = sum;
		}
};

/*
 * Check Node implementation for LogLikelihood quantities
 */

class _LLCheckNode : public _Node
{
	public:
		virtual double messageTo(const std::size_t dest) const
		{
			double v = this->cache();
			const double tv =
				std::tanh(0.5 * this->receivedFrom(dest));

			// tv is the factor contributed to v by the node,
			// which we should leave out. So we need to divide,
			// unless tv equals zero, in which case it was not
			// part of the cached value
			if (tv != 0.0)
				v /= tv;

// 			assert(v > -1.0);
// 			assert(v <  1.0);
			double m = std::log( (1.0+v) / (1.0-v) );
			m = std::min(m,  std::numeric_limits<double>::max());
			m = std::max(m, -std::numeric_limits<double>::max());
			assert( (m <= 0.0) || (m >= 0.0) );
			return m;
/*
			int sign = 1;
			double sumBeta = 0.0;
			for (_Node::const_iterator it = this->begin();
			     it != this->end(); ++it)
			{
				if (it->first == dest)
					continue;
				const double beta = this->receivedFrom(dest);
				sign *= (beta >= 0.0) ? 1 : -1;
				sumBeta += phi0(std::abs(beta));
			}

			return double(sign) * phi0(sumBeta);
*/
		}

	protected:
		virtual void updateCache()
		{
			double prod = 1.0;
			std::size_t zeros = 0u;
			for (_Node::const_iterator it = this->begin();
			     it != this->end(); ++it)
			{
				const double tv = std::tanh(0.5 * it->second);
				// Leave out zero factors, they would not be
				// removable
				assert( (tv <= 0.0) || (tv >= 0.0) );

				if (tv != 0.0)
					prod *= tv;
				else
					++zeros;
				if (zeros > 1u)
				{
					this->cache() = 0.0;
					return;
				}
			}
			assert(zeros <= 1u);
			assert( (prod <= 0.0) || (prod >= 0.0) );
			this->cache() = prod;
		}

	private:
/*
		static double phi0(const double x)
		{
			assert(x >= 0.0);

			if (x == 0.0)
				return 1.0;
			if (x > std::numeric_limits<double>::max())
				return 0.0;

			return -std::log(std::tanh(x/double(2)));
		}
*/
};

template <std::size_t NVariables, std::size_t NChecks>
class LLBeliefPropagator
{
	public:
		LLBeliefPropagator()
		{
			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for (int i = 0; i < int(NVariables); ++i)
				_variables[i].setID(i);

			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for (int i = 0; i < int(NChecks); ++i)
				_checks[i].setID(i);
		}

		template <typename Iterator>
		void init(Iterator first, const Iterator last)
		{
			typedef typename std::iterator_traits<Iterator>::difference_type difference_t;
			assert(std::distance(first, last) == NVariables);
			for (difference_t i = 0; first != last; ++i, ++first)
				_variables[i].init(*first); 

			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for (int i = 0; i < int(NChecks); ++i)
				_checks[i].init();
		}

		void link(const std::size_t variable, const std::size_t check)
		{
			_variables[variable].addLink(&_checks[check]);
			_checks[check].addLink(&_variables[variable]);
		}

		void update()
		{
			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for (int i = 0; i < int(NChecks); ++i)
				_checks[i].calculate();

			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for (int i = 0; i < int(NVariables); ++i)
				_variables[i].calculate();
		}

		void belief(std::bitset<NVariables> &bits) const
		{
			for (std::size_t i = 0; i < NVariables; ++i)
				bits.set(i, _variables[i].bit());
		}

#ifndef NDEBUG
		void print() const
		{
			std::cout << "_BeliefPropagator::print() Check:"
				<< std::endl;
			for (std::size_t i = 0u; i < NChecks; ++i)
				_checks[i].print();

			std::cout << "_BeliefPropagator::print() Variables:"
				<< std::endl;
			for (std::size_t i = 0u; i < NVariables; ++i)
				_variables[i].print();
		}
#endif

	private:
		std::tr1::array<_LLVariableNode, NVariables> _variables;
		std::tr1::array<_LLCheckNode,    NChecks>    _checks;
};

} // end namespace
