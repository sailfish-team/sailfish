/***************************************************************************
 *   Copyright (C) 2007 by F. P. Beekhof                                   *
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

#include <string>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <vector>
#include <iterator>

#include <tr1/array>
#include <tr1/type_traits>

#include <cvmlcpp/base/stl_cstdint.h>

#include <cvmlcpp/math/ContainerOps.h>

#include <cvmlcpp/base/Enums>
#include <cvmlcpp/base/Functors>
#include <cvmlcpp/base/StringTools>

namespace cvmlcpp
{


/**
 * A vector of arbitrary but fixed length.
 */
template <typename T, std::size_t DIMS>
class StaticVector : public std::tr1::array<T, DIMS>
{
	public:
		typedef typename std::tr1::array<T, DIMS>::value_type
								value_type;
		typedef typename std::tr1::array<T, DIMS>::iterator   iterator;
		typedef typename std::tr1::array<T, DIMS>::const_iterator
								const_iterator;

		StaticVector() { }

		template <typename It>
		StaticVector(It first, It last)
		{
			std::copy(first, last, this->begin());
		}

		// Copy-constructor
		template <typename U>
		StaticVector(const U &data)
		{
			this->init(data, std::tr1::is_arithmetic<U>());
		}

		template <typename It>
		void load(It first, It last)
		{
			std::copy(first, last, this->begin());
		}

		template <typename U>
		StaticVector<U, DIMS> convert() const
		{
			return StaticVector<U,DIMS>(*this);
		}

		void clear() { *this = value_type(0); }

		const StaticVector &operator=(const value_type value)
		{
			std::fill(this->begin(), this->end(), value);
			return *this;
		}

		template <typename U>
		const StaticVector &operator=(const U &data)
		{
			this->init(data, std::tr1::is_arithmetic<U>());
			return *this;
		}

		template <typename RHS>
		const StaticVector &operator+=(const RHS &that)
		{
			return detail::doOpIs<StaticVector, std::plus<T> >::
				execute(*this, that);
		}

		template <typename RHS>
		const StaticVector &operator-=(const RHS &that)
		{
			return detail::doOpIs<StaticVector, std::minus<T> >::
				execute(*this, that);
		}


		template <typename RHS>
		const StaticVector &operator*=(const RHS &that)
		{
			return detail::doOpIs<StaticVector, std::multiplies<T> >::
				execute(*this, that);
		}

		template <typename RHS>
		const StaticVector &operator/=(const RHS &that)
		{
			return detail::doOpIs<StaticVector, std::divides<T> >::
				execute(*this, that);
		}

		template <typename RHS>
		const StaticVector &operator%=(const RHS &that)
		{
			return detail::doOpIs<StaticVector, std::modulus<T> >::
				execute(*this, that);
		}

		template <typename U>
		bool operator==(const U &that) const
		{
			return this->equals(that, std::tr1::is_arithmetic<U>());
		}

		template <typename U>
		bool operator!=(const U &that) const
		{
			return !(*this == that);
		}

		const std::string to_string() const
		{
			return cvmlcpp::to_string(this->begin(), this->end());
		}

	protected:
		// Fill with 'value'
		template <typename U>
		void init(const U value, std::tr1::true_type)
		{
			std::fill(this->begin(),this->end(), value_type(value));
		}

		template <typename Container>
		void init(const Container &data, std::tr1::false_type)
		{
			assert(data.size() == this->size());
			std::copy(data.begin(), data.end(), this->begin());
		}

		template <typename U>
		bool equals(const U value, std::tr1::true_type) const
		{
			return (std::find_if(this->begin(), this->end(),
		std::bind2nd(std::not_equal_to<T>(), value)) == this->end());
		}

		template <typename Container>
		bool equals(const Container &data, std::tr1::false_type) const
		{
			assert(data.size() == this->size());
			return std::equal(this->begin(), this->end(),
					  data.begin());
		}
};

template <std::size_t DIMS>
class StaticVector<bool, DIMS>
{
	public:
		typedef bool value_type;
		typedef uint32_t register_type;
		typedef register_type* iterator;
		typedef const register_type * const_iterator;

		// tohex String Length
		static const std::size_t L = (DIMS/4u) + (((DIMS%4u)>0u) ? 1u:0u);

	private:
		static const std::size_t S = 2u * sizeof(register_type);
		static const std::size_t B = 8u * sizeof(register_type);

	public:
		// Nr of raw data elements
		static const std::size_t N = (DIMS/B) + (((DIMS % B)>0u) ? 1u:0u);

		StaticVector() { }

		template <typename Iterator>
		StaticVector(const Iterator first, const Iterator last)
		{
			if (first != last)
				this->load(first, last, *first);
		}

		StaticVector(const std::string hex) { this->set(hex); }

		// Iterators to RAW data
		iterator begin() { return _data; }
		iterator end() { return _data + N; }

		const_iterator begin() const { return _data; }
		const_iterator end() const { return _data + N; }

		void clear()
		{ std::fill(this->begin(), this->end(), register_type(0u)); }

		StaticVector &operator=(const std::string &hex)
		{ this->set(hex); return *this; }

		void set(std::string hex)
		{
			assert(hex.size() >= L);

			to_lower(hex);
			const char *lut = "0123456789abcdef";
			std::size_t c = 0u;

			_data[N-1u] = 0u;
			for (long int j = ((L-1) % S); (j>=0) && (c<L); --j, ++c)
			{
				assert(std::strchr(lut, hex[c]));
				_data[N-1u] |=
					(std::strchr(lut,hex[c])-lut) << (4u*j);
			}
			for (long int i = N-2; i >=0; --i)
			{
				assert(c<L);
				_data[i] = 0u;
				for (long int j = S-1; (j>=0) && (c<L); --j, ++c)
				{
					assert(c<L);
					assert(std::strchr(lut, hex[c]));
					_data[i] |=
					(std::strchr(lut,hex[c])-lut) << (4u*j);
				}
			}
		}

		template <typename Iterator>
		void set(const Iterator first, const Iterator last)
		{
			if (first != last)
				this->load(first, last, *first);
		}

		void setElement(const std::size_t index, const bool value)
		{
			assert(index < DIMS);
			assert(index / B < N);
			if (value)
				_data[index / B] |= (1u << (index % B));
			else
				_data[index / B] &= ~(1u << (index % B));
		}

		const bool operator[] (const std::size_t index) const
		{
			assert(index < DIMS);
			assert(index / B < N);
			return (_data[index / B] >> (index % B)) & 1u;
		}

		const StaticVector operator~() const
		{
			StaticVector that = StaticVector(*this);
			that.flip();
			return that;
		}

		void flip()
		{
			for (std::size_t i = 0u; i < N; ++i)
				_data[i] = ~_data[i];
		}

		void flip(const std::size_t index)
		{
			this->setElement(index, !(*this)[index]);
		}

		std::size_t count() const
		{
			std::size_t cnt	= 0u;
			std::size_t b	= 0u;
			for (std::size_t i = 0u; i < N; ++i)
				for (std::size_t j = 0u; (j<B)&&(b<DIMS); ++j,++b)
					if (_data[i] & (1u << j))
						++cnt;
			return cnt;
		}

		bool odd() const
		{

			return this->count() & 1u;
		}

		bool even() const { return !this->odd(); }

		const StaticVector &operator|=(const StaticVector &that)
		{
			typedef binary_or<register_type> OR;
			std::transform(&this->_data[0], &this->_data[N],
					&that._data[0], &this->_data[0], OR());
			return *this;
		}

		const StaticVector &operator|=(const bool value)
		{
			const register_type mask = value ? ~register_type(0):0;
			typedef binary_or<register_type> OR;
			std::transform(&_data[0], &_data[N], &_data[0],
					std::bind2nd(OR(), mask));
			return *this;
		}

		const StaticVector &operator^=(const StaticVector &that)
		{
			typedef binary_xor<register_type> XOR;
			std::transform(&this->_data[0], &this->_data[N],
					&that._data[0], &this->_data[0], XOR());
			return *this;
		}

		const StaticVector &operator^=(const bool value)
		{
			const register_type mask = value ? ~register_type(0):0;
			typedef binary_xor<register_type> XOR;
			std::transform(&_data[0], &_data[N], &_data[0],
					std::bind2nd(XOR(), mask));
			return *this;
		}

		const StaticVector &operator&=(const StaticVector &that)
		{
			typedef binary_and<register_type> AND;
			std::transform(&this->_data[0], &this->_data[N],
					&that._data[0], &this->_data[0], AND());
			return *this;
		}

		const StaticVector &operator&=(const bool value)
		{
			const register_type mask = value ? ~register_type(0):0;

			typedef binary_and<register_type> AND;
			std::transform(&_data[0], &_data[N], &_data[0],
					std::bind2nd(AND(), mask));
			return *this;
		}

		static std::size_t size() { return DIMS; }

		const std::string to_string() const
		{
			const char lut[] = "0123456789abcdef";
			std::string buf(L, '0');
			std::size_t c = 0u;
			for (long int j = ((L-1) % S); (j>=0) && (c<L); --j, ++c)
				buf[c] = lut[(_data[N-1u] >> (4*j))&0xf];
			for (long int i = N-2; i >= 0; --i)
				for (long int j = S-1; (j>=0) && (c<L); --j,++c)
					buf[c] = lut[(_data[i] >> (4*j))&0xf];

			return buf;
		}

	private:
		register_type _data[N];

		template <typename Iterator>
		void load(Iterator first, Iterator last, bool)
		{
			for (std::size_t i = 0u; i < N; ++i)
			{
				_data[i] = 0u; // Clear
				for (std::size_t j = 0u;// Potentially fill
				     ( (first != last) && (j < B) ); ++j)
				{
					_data[i] |= *first ? (1u << j):0u;
					++first;
				}
			}

			// Clear leftover bits
			for (std::size_t i = DIMS; i < B*N; ++i)
				this->setElement(i, false);
		}

		template <typename Iterator>
		void load(Iterator first, Iterator last, register_type)
		{
			assert(std::size_t(std::distance(first, last)) <= N);
			this->clear();
			std::copy(first, last, &_data[0]);

			// Clear leftover bits
			for (std::size_t i = DIMS; i < B*N; ++i)
				_data[N-1u] &= ~(1u << (i%B));
		}

		template <typename Iterator, typename T>
		void load(Iterator first, Iterator last, T)
		{
			assert(std::distance(first, last) *
				sizeof(typename
				    std::iterator_traits<Iterator>::value_type)
				<= N * sizeof(register_type));

			this->clear();
			std::copy(first, last, reinterpret_cast<
			typename std::iterator_traits<Iterator>::value_type *>
			    (&_data[0]));

			// Clear leftover bits
			for (std::size_t i = DIMS; i < B*N; ++i)
				_data[N-1u] &= ~(1u << (i%B));
		}
};

/**
 * A vector of arbitrary length.
 */
template <class T>
class DynamicVector : public std::vector<T>
{
	public:
		typedef typename std::vector<T>::value_type	value_type;
		typedef typename std::vector<T>::iterator	iterator;
		typedef typename std::vector<T>::const_iterator	const_iterator;

		explicit DynamicVector(const std::size_t DIMS=0u)
			: std::vector<T>(DIMS) { }

		DynamicVector(const_iterator first, const_iterator last)
			: std::vector<T>(first, last) { }

		DynamicVector(const std::size_t DIMS, const T value)
			: std::vector<T>(DIMS, value) { }

		template <typename Container>
		DynamicVector(const Container &data)
			: std::vector<T>(data.begin(), data.end()) { }

		void set(const_iterator first, const_iterator last)
		{
			assert(std::distance(first, last) == this->size());
			std::copy(first, last, this->begin());
		}

		const DynamicVector &operator=(const value_type value)
		{
			std::fill(this->begin(), this->end(), value);
			return *this;
		}

		template <typename U>
		const DynamicVector &operator=(const U &data)
		{
			this->init(data, std::tr1::is_arithmetic<U>());
			return *this;
		}

		template <typename RHS>
		const DynamicVector &operator+=(const RHS &that)
		{
			return detail::doOpIs<DynamicVector, std::plus<T> >::
				execute(*this, that);
		}

		template <typename RHS>
		const DynamicVector &operator-=(const RHS &that)
		{
			return detail::doOpIs<DynamicVector, std::minus<T> >::
				execute(*this, that);
		}


		template <typename RHS>
		const DynamicVector &operator*=(const RHS &that)
		{
			return detail::doOpIs<DynamicVector, std::multiplies<T> >::
				execute(*this, that);
		}

		template <typename RHS>
		const DynamicVector &operator/=(const RHS &that)
		{
			return detail::doOpIs<DynamicVector, std::divides<T> >::
				execute(*this, that);
		}

		template <typename RHS>
		const DynamicVector &operator%=(const RHS &that)
		{
			return detail::doOpIs<DynamicVector, std::modulus<T> >::
				execute(*this, that);
		}

		bool operator==(const DynamicVector &that) const
		{
			return this->equals(that, std::tr1::false_type());
		}

		template <typename U>
		bool operator==(const U &that) const
		{
			return this->equals(that, std::tr1::is_arithmetic<U>());
		}

		bool operator!=(const DynamicVector &that) const
		{
			return !(*this == that);
		}

		const std::string to_string() const
		{
			return cvmlcpp::to_string(this->begin(),this->end());
		}

	private:
		// Fill with 'value'
		template <typename U>
		void init(const U value, std::tr1::true_type)
		{
			std::fill(this->begin(),this->end(), value_type(value));
		}

		template <typename Container>
		void init(const Container &data, std::tr1::false_type)
		{
			this->resize(data.size());
			std::copy(data.begin(), data.end(), this->begin());
		}

		template <typename U>
		bool equals(const U value, std::tr1::true_type) const
		{
			return (std::find_if(this->begin(), this->end(),
		std::bind2nd(std::not_equal_to<T>(), value)) == this->end());
		}

		template <typename Container>
		bool equals(const Container &data, std::tr1::false_type) const
		{
			assert(data.size() == this->size());
			return std::equal(this->begin(), this->end(),
					  data.begin());
		}
};
/*
template <>
class DynamicVector<bool>
{
	public:
		typedef bool value_type;
		typedef uint32_t register_type;
		typedef register_type * iterator;
		typedef const register_type * const_iterator;

	private:
		static const std::size_t S = 2u * sizeof(register_type);
		static const std::size_t B = 8u * sizeof(register_type);

	public:

		DynamicVector() { }

		template <typename Iterator>
		DynamicVector(const Iterator first, const Iterator last)
		{
			if (first != last)
				this->load(first, last, *first);
		}

		// lower-case only; no UPPER_CASE!!!
		DynamicVector(const std::string &hex) { this->set(hex); }

		// Iterators to RAW data
		iterator begin() { return _data; }
		iterator end() { return _data + N; }

		const_iterator begin() const { return _data; }
		const_iterator end() const { return _data + N; }

		void clear()
		{ std::fill(this->begin(), this->end(), register_type(0u)); }

		DynamicVector &operator=(const std::string &hex)
		{ this->set(hex); return *this; }

		// lower-case only; no UPPER_CASE!!!
		void set(const std::string &hex)
		{
		// tohex String Length
			static const std::size_t L = (DIMS/4u) +
						(((DIMS%4u)>0u) ? 1u:0u);
			assert(hex.size() >= L);

			const char *lut = "0123456789abcdef";
			std::size_t c = 0u;

			_data[N-1u] = 0u;
			for (long int j = ((L-1) % S); (j>=0) && (c<L); --j, ++c)
			{
				assert(std::strchr(lut, hex[c]));
				_data[N-1u] |=
					(std::strchr(lut,hex[c])-lut) << (4u*j);
			}
			for (long int i = N-2; i >=0; --i)
			{
				assert(c<L);
				_data[i] = 0u;
				for (long int j = S-1; (j>=0) && (c<L); --j, ++c)
				{
					assert(c<L);
					assert(std::strchr(lut, hex[c]));
					_data[i] |=
					(std::strchr(lut,hex[c])-lut) << (4u*j);
				}
			}
		}

		template <typename Iterator>
		void set(const Iterator first, const Iterator last)
		{
			if (first != last)
				this->load(first, last, *first);
		}

		void setElement(const std::size_t index, const bool value)
		{
			assert(index < DIMS);
			assert(index / B < N);
			if (value)
				_data[index / B] |= (1u << (index % B));
			else
				_data[index / B] &= ~(1u << (index % B));
		}

		const bool operator[] (const std::size_t index) const
		{
			assert(index < DIMS);
			assert(index / B < N);
			return (_data[index / B] >> (index % B)) & 1u;
		}

		const DynamicVector operator~() const
		{
			DynamicVector that = *this;
			that.flip();
			return that;
		}

		void flip()
		{
			for (std::size_t i = 0u; i < N; ++i)
				_data[i] = ~_data[i];
		}

		void flip(const std::size_t index)
		{
			this->setElement(index, !(*this)[index]);
		}

		std::size_t count() const
		{
			std::size_t cnt	= 0u;
			std::size_t b	= 0u;
			for (std::size_t i = 0u; i < N; ++i)
				for (std::size_t j = 0u; (j<B)&&(b<DIMS); ++j,++b)
					if (_data[i] & (1u << j))
						++cnt;
			return cnt;
		}

		bool odd() const
		{

			return this->count() & 1u;
		}

		bool even() const { return !this->odd(); }

		const DynamicVector &operator|=(const DynamicVector &that)
		{
			typedef Functors::binary_or<register_type> OR;
			std::transform(&this->_data[0], &this->_data[N],
					&that._data[0], &this->_data[0], OR());
			return *this;
		}

		const DynamicVector &operator|=(const bool value)
		{
			const register_type mask = value ? ~register_type(0):0;
			typedef Functors::binary_or<register_type> OR;
			std::transform(&_data[0], &_data[N], &_data[0],
					std::bind2nd(OR(), mask));
			return *this;
		}

		const DynamicVector &operator^=(const DynamicVector &that)
		{
			typedef Functors::binary_xor<register_type> XOR;
			std::transform(&this->_data[0], &this->_data[N],
					&that._data[0], &this->_data[0], XOR());
			return *this;
		}

		const DynamicVector &operator^=(const bool value)
		{
			const register_type mask = value ? ~register_type(0):0;
			typedef Functors::binary_xor<register_type> XOR;
			std::transform(&_data[0], &_data[N], &_data[0],
					std::bind2nd(XOR(), mask));
			return *this;
		}

		const DynamicVector &operator&=(const DynamicVector &that)
		{
			typedef Functors::binary_and<register_type> AND;
			std::transform(&this->_data[0], &this->_data[N],
					&that._data[0], &this->_data[0], AND());
			return *this;
		}

		const DynamicVector &operator&=(const bool value)
		{
			const register_type mask = value ? ~register_type(0):0;

			typedef Functors::binary_and<register_type> AND;
			std::transform(&_data[0], &_data[N], &_data[0],
					std::bind2nd(AND(), mask));
			return *this;
		}

		static std::size_t size() { return DIMS; }

		const std::string to_string() const
		{
			const char lut[] = "0123456789abcdef";
			std::string buf(L, '0');
			std::size_t c = 0u;
			for (long int j = ((L-1) % S); (j>=0) && (c<L); --j, ++c)
				buf[c] = lut[(_data[N-1u] >> (4*j))&0xf];
			for (long int i = N-2; i >= 0; --i)
				for (long int j = S-1; (j>=0) && (c<L); --j,++c)
					buf[c] = lut[(_data[i] >> (4*j))&0xf];

			return buf;
		}

	private:
		std::vector<register_type> _data;

		template <typename Iterator>
		void load(Iterator first, Iterator last, bool)
		{
			for (std::size_t i = 0u; i < N; ++i)
			{
				_data[i] = 0u; // Clear
				for (std::size_t j = 0u;// Potentially fill
				     ( (first != last) && (j < B) ); ++j)
				{
					_data[i] |= *first ? (1u << j):0u;
					++first;
				}
			}

			// Clear leftover bits
			for (std::size_t i = DIMS; i < B*N; ++i)
				this->setElement(i, false);
		}

		template <typename Iterator>
		void load(Iterator first, Iterator last, register_type)
		{
			assert(std::size_t(std::distance(first, last)) <= N);
			this->clear();
			std::copy(first, last, &_data[0]);

			// Clear leftover bits
			for (std::size_t i = DIMS; i < B*N; ++i)
				_data[N-1u] &= ~(1u << (i%B));
		}

		template <typename Iterator, typename T>
		void load(Iterator first, Iterator last, T)
		{
			assert(std::distance(first, last) *
				sizeof(typename
				    std::iterator_traits<Iterator>::value_type)
				<= N * sizeof(register_type));

			this->clear();
			std::copy(first, last,
			(typename std::iterator_traits<Iterator>::value_type *)
			    (&_data[0]));

			// Clear leftover bits
			for (std::size_t i = DIMS; i < B*N; ++i)
				_data[N-1u] &= ~(1u << (i%B));
		}

};
*/

/*
template <typename T, std::size_t DIMS, typename U>
const bool operator==(const U &lhs, const StaticVector<T, DIMS> &rhs)
{
	return rhs == lhs;
}

template <typename T, std::size_t DIMS>
const bool operator==(const T &lhs, const StaticVector<T, DIMS> &rhs)
{
	return rhs == lhs;
}
*/

/*
template <typename T, std::size_t DIMS, typename U>
const bool operator!=(const StaticVector<T, DIMS> &lhs, const U &rhs)
{
	return !(lhs == rhs);
}

template <typename T, std::size_t DIMS, typename U>
const bool operator!=(const U &lhs, const StaticVector<T, DIMS> &rhs)
{
	return !(rhs == lhs);
}
*/

template <typename T, std::size_t DIMS, typename U, typename Op>
const bool strictComp(const StaticVector<T, DIMS> &lhs, const U &rhs,
		Op op, std::tr1::false_type)
{
	assert(lhs.size() == rhs.size());
	for (std::size_t i = 0u; i < DIMS; ++i)
	{
		if (op(lhs[i], rhs[i]))
			return true;
		if (lhs[i] != rhs[i])
			return false;
	}

	return false;
}

template <typename T, std::size_t DIMS, typename U, typename Op>
const bool strictComp(const StaticVector<T, DIMS> &lhs, const U &rhs,
		Op op, std::tr1::true_type)
{
	for (std::size_t i = 0u; i < DIMS; ++i)
	{
		if (op(lhs[i], rhs))
			return true;
		if (lhs[i] != rhs)
			return false;
	}

	return false;
}

// Less
template <typename T, std::size_t DIMS>
const bool operator<(const StaticVector<T, DIMS> &lhs,
			const StaticVector<T, DIMS> &rhs)
{
	return strictComp(lhs, rhs, std::less<T>(),
			  std::tr1::false_type());
}

template <typename T, std::size_t DIMS>
const bool operator<(const StaticVector<T, DIMS> &lhs,
			const DynamicVector<T> &rhs)
{
	return strictComp(lhs, rhs, std::less<T>(),
			  std::tr1::false_type());
}

template <typename T, std::size_t DIMS>
const bool operator<(const StaticVector<T, DIMS> &lhs, const T &rhs)
{
	return strictComp(lhs, rhs, std::less<T>(),
			  std::tr1::true_type());
}

template <typename T, std::size_t DIMS, typename U>
const bool operator<(const StaticVector<T, DIMS> &lhs, const U &rhs)
{
	return strictComp(lhs, rhs, std::less<T>(),
			  std::tr1::is_arithmetic<U>());
}

// Greater
template <typename T, typename U, std::size_t DIMS>
const bool operator>(const StaticVector<T, DIMS> &lhs,
			const StaticVector<U, DIMS> &rhs)
{
	return strictComp(lhs, rhs, std::greater<T>(),
			  std::tr1::false_type());
}

template <typename T, typename U, std::size_t DIMS>
const bool operator>(const StaticVector<T, DIMS> &lhs,
			const DynamicVector<U> &rhs)
{
	return strictComp(lhs, rhs, std::greater<T>(),
			  std::tr1::false_type());
}

template <typename T, std::size_t DIMS>
const bool operator>(const StaticVector<T, DIMS> &lhs, const T &rhs)
{
	return strictComp(lhs, rhs, std::greater<T>(),
			  std::tr1::true_type());
}

template <typename T, std::size_t DIMS, typename U>
const bool operator>(const StaticVector<T, DIMS> &lhs, const U &rhs)
{
	return strictComp(lhs, rhs, std::greater<T>(),
			  std::tr1::is_arithmetic<U>());
}

template <typename T, std::size_t DIMS, typename U, typename Op>
const bool comp(const StaticVector<T, DIMS> &lhs, const U &rhs,
		Op op, std::tr1::false_type)
{
	assert(lhs.size() == rhs.size());
	for (std::size_t i = 0u; i < DIMS; ++i)
		if (!op(lhs[i], rhs[i]))
			return false;

	return true;
}

template <typename T, std::size_t DIMS, typename U, typename Op>
const bool comp(const StaticVector<T, DIMS> &lhs, const U &rhs,
		Op op, std::tr1::true_type)
{
	for (std::size_t i = 0u; i < DIMS; ++i)
		if (!op(lhs[i], rhs))
			return false;

	return true;
}

// Less-equal
template <typename T, typename U, std::size_t DIMS>
const bool operator<=(const StaticVector<T, DIMS> &lhs,
			const StaticVector<U, DIMS> &rhs)
{
	return comp(lhs, rhs, std::less_equal<T>(), std::tr1::false_type());
}

template <typename T, typename U, std::size_t DIMS>
const bool operator<=(const StaticVector<T, DIMS> &lhs,
			const DynamicVector<U> &rhs)
{
	return comp(lhs, rhs, std::less_equal<T>(), std::tr1::false_type());
}

template <typename T, std::size_t DIMS>
const bool operator<=(const StaticVector<T, DIMS> &lhs, const T &rhs)
{
	return comp(lhs, rhs, std::less_equal<T>(), std::tr1::true_type());
}

template <typename T, std::size_t DIMS, typename U>
const bool operator<=(const StaticVector<T, DIMS> &lhs, const U &rhs)
{
	return comp(lhs, rhs, std::less_equal<T>(),
		    std::tr1::is_arithmetic<U>());
}

// Greater-equal
template <typename T, std::size_t DIMS>
const bool operator>=(const StaticVector<T, DIMS> &lhs,
			const StaticVector<T, DIMS> &rhs)
{
	return comp(lhs, rhs, std::greater_equal<T>(), std::tr1::false_type());
}

template <typename T, std::size_t DIMS>
const bool operator>=(const StaticVector<T, DIMS> &lhs,
			const DynamicVector<T> &rhs)
{
	return comp(lhs, rhs, std::greater_equal<T>(), std::tr1::false_type());
}

template <typename T, std::size_t DIMS>
const bool operator>=(const StaticVector<T, DIMS> &lhs, const T &rhs)
{
	return comp(lhs, rhs, std::greater_equal<T>(), std::tr1::true_type());
}

template <typename T, std::size_t DIMS, typename U>
const bool operator>=(const StaticVector<T, DIMS> &lhs, const U &rhs)
{
	return comp(lhs, rhs, std::greater_equal<T>(),
			std::tr1::is_arithmetic<U>());
}


template <typename T, typename TLHS, typename TRHS>
const StaticVector<T, 3u>
_crossProduct(const TLHS &lhs, const TRHS &rhs, T)
{
	assert(lhs.size() == 3u);
	assert(rhs.size() == 3u);
	StaticVector<T, 3u> v;

	v[X] = lhs[Y] * rhs[Z] - lhs[Z] * rhs[Y];
	v[Y] = lhs[Z] * rhs[X] - lhs[X] * rhs[Z];
	v[Z] = lhs[X] * rhs[Y] - lhs[Y] * rhs[X];

	return v;
}

template <typename T>
const StaticVector<T, 3u>
crossProduct(const StaticVector<T, 3u> &lhs, const StaticVector<T, 3u> &rhs)
{
	return _crossProduct(lhs, rhs, T());
}

template <typename T>
const StaticVector<T, 3u>
crossProduct(const StaticVector<T, 3u> &lhs, const DynamicVector<T> &rhs)
{
	return _crossProduct(lhs, rhs, T());
}

template <typename T>
const StaticVector<T, 3u>
crossProduct(const DynamicVector<T> &lhs, const StaticVector<T, 3u> &rhs)
{
	assert(rhs.size() == 3u);
	return _crossProduct(lhs, rhs, T());
}


template <typename T, std::size_t DIMS, typename U>
T dotProduct(const StaticVector<T, DIMS> &lhs, const U &rhs)
{
	assert(lhs.size() == rhs.size());
	T rv = T(0);

	typename StaticVector<T, DIMS>::const_iterator l = lhs.begin();
	typename U::const_iterator r = rhs.begin();

	for (/* nop */; l != lhs.end(); ++l, ++r)
		rv += *l * *r;

	return rv;
}

template <typename T, std::size_t DIMS>
const StaticVector<T, DIMS> operator+(const StaticVector<T,DIMS> &lhs,
					const T &rhs)
{
	return detail::doOp<StaticVector<T, DIMS>, std::plus<T> >::execute(lhs, rhs);
}

template <typename T, std::size_t DIMS>
const StaticVector<T, DIMS> operator+(const T &lhs,
					const StaticVector<T,DIMS> &rhs)
{
	return detail::doOp<StaticVector<T, DIMS>, std::plus<T> >::execute(lhs, rhs);
}

template <typename T, std::size_t DIMS>
const StaticVector<T, DIMS> operator+(const StaticVector<T,DIMS> &lhs,
					const StaticVector<T,DIMS> &rhs)
{
	return detail::doOp<StaticVector<T, DIMS>, std::plus<T> >::execute(lhs, rhs);
}

template <typename T, std::size_t DIMS>
const StaticVector<T, DIMS> operator-(const T &lhs,
					const StaticVector<T,DIMS> &rhs)
{
	return detail::doOp<StaticVector<T, DIMS>, std::minus<T> >::execute(lhs, rhs);
}

template <typename T, std::size_t DIMS>
const StaticVector<T, DIMS> operator-(const StaticVector<T,DIMS> &lhs,
					const T &rhs)
{
	return detail::doOp<StaticVector<T, DIMS>, std::minus<T> >::execute(lhs, rhs);
}

template <typename T, std::size_t DIMS>
const StaticVector<T, DIMS> operator-(const StaticVector<T, DIMS> &lhs,
					const StaticVector<T, DIMS> &rhs)
{
	return detail::doOp<StaticVector<T, DIMS>, std::minus<T> >::execute(lhs, rhs);
}

template <typename T, std::size_t DIMS>
const StaticVector<T, DIMS> operator*(const StaticVector<T,DIMS> &lhs,
					const T &rhs)
{
	return detail::doOp<StaticVector<T, DIMS>, std::multiplies<T> >::
		execute(lhs, rhs);
}

template <typename T, std::size_t DIMS>
const StaticVector<T, DIMS> operator*(const T &lhs,
					const StaticVector<T,DIMS> &rhs)
{
	return detail::doOp<StaticVector<T, DIMS>, std::multiplies<T> >::
		execute(lhs, rhs);
}

template <typename T, std::size_t DIMS>
const StaticVector<T, DIMS> operator*(const StaticVector<T,DIMS> &lhs,
					const StaticVector<T,DIMS> &rhs)
{
	return detail::doOp<StaticVector<T, DIMS>, std::multiplies<T> >::
			execute(lhs, rhs);
}

template <typename T, std::size_t DIMS>
const StaticVector<T, DIMS> operator/(const StaticVector<T,DIMS> &lhs,
					const T &rhs)
{
	return detail::doOp<StaticVector<T, DIMS>, std::divides<T> >::
		execute(lhs, rhs);
}

template <typename T, std::size_t DIMS>
const StaticVector<T, DIMS> operator/(const T &lhs,
					const StaticVector<T, DIMS> &rhs)
{
	return detail::doOp<StaticVector<T, DIMS>, std::divides<T> >::
		execute(lhs, rhs);
}

template <typename T, std::size_t DIMS>
const StaticVector<T, DIMS> operator/(const StaticVector<T,DIMS> &lhs,
					const StaticVector<T,DIMS> &rhs)
{
	return detail::doOp<StaticVector<T, DIMS>, std::divides<T> >::
			execute(lhs, rhs);
}

template <typename T, std::size_t DIMS>
const StaticVector<T, DIMS> operator%(const StaticVector<T,DIMS> &lhs,
					const T &rhs)
{
// 	typedef typename ValueType<StaticVector<T, DIMS> >::value_type VT;
	return detail::doOp<StaticVector<T, DIMS>, std::modulus<T> >::
		execute(lhs, rhs);
}

template <typename T, std::size_t DIMS>
const StaticVector<T, DIMS> operator%(const T &lhs,
					const StaticVector<T, DIMS> &rhs)
{
	return detail::doOp<StaticVector<T, DIMS>, std::modulus<T> >::
		execute(lhs, rhs);
}

template <typename T, std::size_t DIMS>
const StaticVector<T, DIMS> operator%(const StaticVector<T,DIMS> &lhs,
					const StaticVector<T,DIMS> &rhs)
{
	return detail::doOp<StaticVector<T, DIMS>, std::modulus<T> >::
			execute(lhs, rhs);
}

template <std::size_t DIMS>
const StaticVector<bool, DIMS> operator&&(const StaticVector<bool, DIMS> &lhs,
					const bool &rhs)
{
	return detail::doOp<StaticVector<bool, DIMS>, std::logical_and<bool> >::
			execute(lhs, rhs);
}

template <std::size_t DIMS>
const StaticVector<bool, DIMS> operator&&(const bool &lhs,
					const StaticVector<bool, DIMS> &rhs)
{
	return detail::doOp<StaticVector<bool, DIMS>, std::logical_and<bool> >::
			execute(lhs, rhs);
}

template <std::size_t DIMS>
const StaticVector<bool, DIMS> operator&&(const StaticVector<bool, DIMS> &lhs,
					  const StaticVector<bool, DIMS> &rhs)
{
	return detail::doOp<StaticVector<bool, DIMS>, std::logical_and<bool> >::
			execute(lhs, rhs);
}

template <typename T, std::size_t DIMS>
const StaticVector<T, DIMS> operator&(const StaticVector<T, DIMS> &lhs,
					 const StaticVector<T, DIMS> &rhs)
{
	StaticVector<T, DIMS> v(lhs);
	v &= rhs;
	return v;
}

template <typename T, std::size_t DIMS>
const StaticVector<T, DIMS> operator|(const StaticVector<T, DIMS> &lhs,
					 const StaticVector<T, DIMS> &rhs)
{
	StaticVector<T, DIMS> v(lhs);
	v |= rhs;
	return v;
}

template <typename T, std::size_t DIMS>
const StaticVector<T, DIMS> operator^(const StaticVector<T, DIMS> &lhs,
					 const StaticVector<T, DIMS> &rhs)
{
	StaticVector<T, DIMS> v(lhs);
	v ^= rhs;
	return v;
}


/*
 * DynamicVector Operators
 */

template <typename T, typename U>
const bool operator<(const DynamicVector<T> &lhs, const U &rhs)
{
	assert(lhs.size() == rhs.size());
	for (std::size_t i = 0u; i < lhs.size(); ++i)
	{
		if ( lhs[i] < rhs[i])
			return true;
		if ( /*( lhs[i] > rhs[i]) || */
			( lhs[i] != rhs[i] ) )
			return false;
		assert(lhs[i] == rhs[i]);
	}

	return false;
}

template <typename T, typename U>
const bool operator>(const DynamicVector<T> &lhs, const U &rhs)
{
	assert(lhs.size() == rhs.size());
	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		if ( lhs[i] > rhs[i])
			return true;
		if ( /*( lhs[i] < rhs[i]) || */
			( lhs[i] != rhs[i] ) )
			return false;
		assert(lhs[i] == rhs[i]);
	}

	return false;
}

template <typename T>
const bool operator<(const DynamicVector<T> &lhs, const T &rhs)
{
	for (std::size_t i = 0u; i < lhs.size(); ++i)
	{
		if (lhs[i] < rhs)
			return true;
		if ( /*( lhs[i] > rhs[i]) || */
			( lhs[i] != rhs ) )
			return false;
		assert(lhs[i] == rhs);
	}

	return false;
}

template <typename T>
const bool operator>(const DynamicVector<T> &lhs, const T &rhs)
{
	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		if ( lhs[i] > rhs)
			return true;
		if ( /*( lhs[i] < rhs[i]) || */
			( lhs[i] != rhs) )
			return false;
		assert(lhs[i] == rhs);
	}

	return false;
}

template <typename T, typename U>
const bool operator<=(const DynamicVector<T> &lhs, const U &rhs)
{
	assert(lhs.size() == rhs.size());
	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		if ( lhs[i] < rhs[i])
			return true;
		if ( lhs[i] > rhs[i])
			return false;
		assert(lhs[i] == rhs[i]);
	}

	return true;
}

template <typename T, typename U>
const bool operator>=(const DynamicVector<T> &lhs, const U &rhs)
{
	assert(lhs.size() == rhs.size());
	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		if ( lhs[i] > rhs[i])
			return true;
		if ( lhs[i] < rhs[i])
			return false;
		assert(lhs[i] == rhs[i]);
	}

	return true;
}

template <typename T>
const bool operator<=(const DynamicVector<T> &lhs, const T &rhs)
{
	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		if (lhs[i] < rhs)
			return true;
		if (lhs[i] > rhs)
			return false;
		assert(lhs[i] == rhs);
	}

	return true;
}

template <typename T>
const bool operator>=(const DynamicVector<T> &lhs, const T &rhs)
{
	for (std::size_t i = 0; i < lhs.size(); ++i)
	{
		if (lhs[i] > rhs)
			return true;
		if (lhs[i] < rhs)
			return false;
		assert(lhs[i] == rhs);
	}

	return true;
}

template <typename T, typename U>
const bool operator==(const U &lhs, const DynamicVector<T> &rhs)
{
	return rhs == lhs;
}

template <typename T, typename U>
const bool operator!=(const DynamicVector<T> &lhs, const U &rhs)
{
	return !(lhs == rhs);
}

template <typename T, typename U>
const bool operator!=(const U &lhs, const DynamicVector<T> &rhs)
{
	return !(rhs == lhs);
}

template <typename T, typename U>
const DynamicVector<T> cross(const DynamicVector<T> &lhs, const U &rhs)
{
	assert(lhs.size() == 3u);
	assert(rhs.size() == 3u);
	DynamicVector<T> v(3u);

	v[X]	= lhs[Y] * rhs[Z] - lhs[Z] * rhs[Y];
	v[Y]	= lhs[Z] * rhs[X] - lhs[X] * rhs[Z];
	v[Z]	= lhs[X] * rhs[Y] - lhs[Y] * rhs[X];

	return v;
}

template <typename T, typename U>
const T dotProduct(const DynamicVector<T> &lhs, const U &rhs)
{
	assert(lhs.size() == rhs.size());
	T res = T(0);

	typename DynamicVector<T>::const_iterator i = lhs.begin();
	typename DynamicVector<T>::const_iterator j = rhs.begin();
	while (i != lhs.end())
	{
		res += *i * *j;
		++i;
		++j;
	}

	return res;
}

template <typename T, class RHS>
const DynamicVector<T> operator+(const DynamicVector<T> &lhs, const RHS &rhs)
{
	typedef typename ValueType<DynamicVector<T> >::value_type value_type;
	return detail::doOp<DynamicVector<T>, std::plus<value_type> >::
			execute(lhs, rhs);
}

template <typename T, class LHS>
const DynamicVector<T> operator+(const LHS &lhs, const DynamicVector<T> &rhs)
{
	typedef typename ValueType<DynamicVector<T> >::value_type value_type;
	return detail::doOp<DynamicVector<T>, std::plus<value_type> >::
			execute(lhs, rhs);
}

template <typename T>
const DynamicVector<T> operator+(const DynamicVector<T> &lhs,
				 const DynamicVector<T> &rhs)
{
	typedef typename ValueType<DynamicVector<T> >::value_type value_type;
	return detail::doOp<DynamicVector<T>, std::plus<value_type> >::
			execute(lhs, rhs);
}

template <typename T, class RHS>
const DynamicVector<T> operator-(const DynamicVector<T> &lhs, const RHS &rhs)
{
	typedef typename ValueType<DynamicVector<T> >::value_type value_type;
	return detail::doOp<DynamicVector<T>, std::minus<value_type> >::
			execute(lhs, rhs);
}

template <typename T, class LHS>
const DynamicVector<T> operator-(const LHS &lhs, const DynamicVector<T> &rhs)
{
	typedef typename ValueType<DynamicVector<T> >::value_type value_type;
	return detail::doOp<DynamicVector<T>, std::minus<value_type> >::
			execute(lhs, rhs);
}

template <typename T>
const DynamicVector<T> operator-(const DynamicVector<T> &lhs,
				 const DynamicVector<T> &rhs)
{
	typedef typename ValueType<DynamicVector<T> >::value_type value_type;
	return detail::doOp<DynamicVector<T>, std::minus<value_type> >::
			execute(lhs, rhs);
}

template <typename T, class RHS>
const DynamicVector<T> operator*(const DynamicVector<T> &lhs, const RHS &rhs)
{
	typedef typename ValueType<DynamicVector<T> >::value_type value_type;
	return detail::doOp<DynamicVector<T>, std::multiplies<value_type> >::
		execute(lhs, rhs);
}

template <typename T, class LHS>
const DynamicVector<T> operator*(const LHS &lhs, const DynamicVector<T> &rhs)
{
	typedef typename ValueType<DynamicVector<T> >::value_type value_type;
	return detail::doOp<DynamicVector<T>, std::multiplies<value_type> >::
			execute(lhs, rhs);
}

template <typename T>
const DynamicVector<T> operator*(const DynamicVector<T> &lhs,
				 const DynamicVector<T> &rhs)
{
	typedef typename ValueType<DynamicVector<T> >::value_type value_type;
	return detail::doOp<DynamicVector<T>, std::multiplies<value_type> >::
			execute(lhs, rhs);
}

template <typename T, class RHS>
const DynamicVector<T> operator/(const DynamicVector<T> &lhs, const RHS &rhs)
{
	typedef typename ValueType<DynamicVector<T> >::value_type value_type;
	return detail::doOp<DynamicVector<T>, std::divides<value_type> >::
		execute(lhs, rhs);
}

template <typename T, class LHS>
const DynamicVector<T> operator/(const LHS &lhs, const DynamicVector<T> &rhs)
{
	typedef typename ValueType<DynamicVector<T> >::value_type value_type;
	return detail::doOp<DynamicVector<T>, std::divides<value_type> >::
			execute(lhs, rhs);
}

template <typename T>
const DynamicVector<T> operator/(const DynamicVector<T> &lhs,
				 const DynamicVector<T> &rhs)
{
	typedef typename ValueType<DynamicVector<T> >::value_type value_type;
	return detail::doOp<DynamicVector<T>, std::divides<value_type> >::
			execute(lhs, rhs);
}

template <typename T, class RHS>
const DynamicVector<T> operator%(const DynamicVector<T> &lhs, const RHS &rhs)
{
	typedef typename ValueType<DynamicVector<T> >::value_type value_type;
	return detail::doOp<DynamicVector<T>, std::modulus<value_type> >::
		execute(lhs, rhs);
}

template <typename T, class LHS>
const DynamicVector<T> operator%(const LHS &lhs, const DynamicVector<T> &rhs)
{
	typedef typename ValueType<DynamicVector<T> >::value_type value_type;
	return detail::doOp<DynamicVector<T>, std::modulus<value_type> >::
			execute(lhs, rhs);
}

template <typename T>
const DynamicVector<T> operator%(const DynamicVector<T> &lhs,
				 const DynamicVector<T> &rhs)
{
	typedef typename ValueType<DynamicVector<T> >::value_type value_type;
	return detail::doOp<DynamicVector<T>, std::modulus<value_type> >::
			execute(lhs, rhs);
}

template <class T, std::size_t DIMS>
class StaticSortedVector
{
	public:
		typedef T		value_type;
		typedef const T*	const_iterator;

		StaticSortedVector() { }

		StaticSortedVector(const_iterator first, const_iterator last)
		{
			assert(std::size_t(std::distance(first, last)) == this->size());
			std::copy(first, last, this->begin());
			std::sort(this->begin(), this->end());
		}

		template <class Container>
		StaticSortedVector(const Container &that)
		{
			assert(that.size() == this->size());
			std::copy(that.begin(), that.end(), this->begin());
			std::sort(this->begin(), this->end());
		}

		template <class Container>
		const StaticSortedVector &operator=(const Container &that)
		{
			assert(that.size() == this->size());
			std::copy(that.begin(), that.end(), this->begin());
			std::sort(this->begin(), this->end());
			return *this;
		}

		const_iterator begin() const { return _data; }
		const_iterator end() const { return _data + this->size(); }

		template <typename It>
		void set(It first, It last)
		{
			assert(std::size_t(std::distance(first, last)) == this->size());
			std::copy(first, last, this->begin());
			std::sort(this->begin(), this->end());
		}

		const value_type operator[] (std::size_t index) const
		{
			assert (index < this->size());
			return _data[index];
		}

		bool has(const value_type value) const
		{
			return std::binary_search(this->begin(), this->end(),
						  value) == this->end();
		}

		static std::size_t size() { return DIMS; }

		const std::string to_string() const
		{
			return cvmlcpp::to_string(this->begin(),this->end());
		}

	private:
		typedef T* iterator;
		iterator begin() { return _data; }
		iterator end() { return _data + this->size(); }
		T _data[DIMS];
};

template <typename T>
class DynamicSortedVector
{
	public:
		typedef T value_type;
		typedef typename std::vector<T>::const_iterator	const_iterator;

		DynamicSortedVector(const_iterator first, const_iterator last) :
			_data(first, last)
		{
			std::sort(this->begin(), this->end());
			assert(std::unique(this->begin(), this->end())
				== this->end());
		}

		const_iterator begin() const { return _data.begin(); }
		const_iterator end() const { return _data.end(); }

		const T operator[] (std::size_t index) const
		{
			assert (index < this->size());
			return _data[index];
		}

		bool has(const value_type value) const
		{
			return std::binary_search(this->begin(), this->end(),
						  value) == this->end();
		}

		template <typename It>
		void set(It first, It last)
		{
			assert(std::distance(first, last) == this->size());
			std::copy(first, last, this->begin());
			std::sort(this->begin(), this->end());
		}

		std::size_t size() const { return _data.size(); }

		const std::string to_string() const
		{
			return cvmlcpp::to_string(this->begin(), this->end());
		}

	private:
		typedef typename std::vector<T>::iterator	iterator;
		iterator begin() { return _data.begin(); }
		iterator end() { return _data.end(); }
		std::vector<T> _data;
};

template <typename T, std::size_t DIMS, typename U>
const bool operator==(const StaticSortedVector<T, DIMS> &lhs, const U &rhs)
{
	if (lhs.size() != rhs.size())
		return false;
	return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <typename T, std::size_t DIMS, typename U>
const bool operator!=(const StaticSortedVector<T, DIMS> &lhs, const U &rhs)
{
	return !(lhs == rhs);
}

template <typename T, std::size_t DIMS, typename U>
const bool operator<(const StaticSortedVector<T, DIMS> &lhs, const U &rhs)
{
	assert(lhs.size() == rhs.size());
	for (std::size_t i = 0u; i < DIMS; ++i)
	{
		if (lhs[i] < rhs[i])
			return true;
		if ( /*(lhs[i] > rhs[i]) || */
			(lhs[i] != rhs[i] ) )
			return false;
		assert(lhs[i] == rhs[i]);
	}

	return false;
}

template <typename T, std::size_t DIMS, typename U>
const bool operator>(const StaticSortedVector<T, DIMS> &lhs, const U &rhs)
{
	assert(lhs.size() == rhs.size());
	for (std::size_t i = 0u; i < DIMS; ++i)
	{
		if ( lhs[i] > rhs[i])
			return true;
		if ( /*(lhs[i] < rhs[i]) || */
			(lhs[i] != rhs[i] ) )
			return false;
		assert(lhs[i] == rhs[i]);
	}

	return false;
}

} // namespace cvmlcpp

template <typename T, std::size_t DIMS>
std::ostream & operator<<(std::ostream &o, const cvmlcpp::StaticVector<T, DIMS> &v)
{
	o << v.to_string();
	return o;
}
