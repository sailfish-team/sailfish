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

#include <algorithm>
#include <functional>
#include <iostream>
#include <complex>
#include <climits>
#include <limits>
#include <cassert>

#ifdef __GXX_EXPERIMENTAL_CXX0X__
#include <type_traits>
#else
#include <type_traits>
#endif

/*
 * Functors required in Math need to be defined first
 */

namespace cvmlcpp
{

/*
 * Binary operators
 */

template <typename T>
struct binary_and : public std::binary_function<T, T, T>
{
	typedef T value_type;
	typedef T result_type;

	T operator() (const T &lhs, const T &rhs) const
	{ return lhs & rhs; }
};

template <typename T>
struct binary_or : public std::binary_function<T, T, T>
{
	typedef T value_type;
	typedef T result_type;

	T operator() (const T &lhs, const T &rhs) const
	{ return lhs | rhs; }
};

template <typename T>
struct binary_xor : public std::binary_function<T, T, T>
{
	typedef T value_type;
	typedef T result_type;

	T operator() (const T &lhs, const T &rhs) const
	{ return lhs ^ rhs; }
};

template <typename T>
struct binary_not : public std::unary_function<T, T>
{
	typedef T value_type;
	typedef T result_type;

	T operator() (const T &o) const
	{ return ~o; }
};

template <typename TSrc, typename TDest>
struct StaticCaster : public std::unary_function<TSrc, TDest>
{
	TDest operator()(const TSrc &src) const
	{
		return static_cast<TDest>(src);
	}
};

namespace detail
{

template <typename T, bool is_integer>
struct RoughlyEqualToScalar
{
	static bool roughly_equal_to (const T &lhs, const T &rhs)
	{
		assert(!std::numeric_limits<T>::is_integer);
		using std::max; using std::pow; using std::abs;

		// last two digits of largest number
		const T epsilon = max(abs(lhs), abs(rhs)) * pow(T(2), -T(std::numeric_limits<T>::digits-2));
		assert(epsilon >= T(0));
		assert(abs(lhs) + epsilon >= lhs);
		assert(abs(rhs) + epsilon >= rhs);
		//std::cerr << lhs << " " << rhs << " " << epsilon << " " << std::numeric_limits<T>::epsilon() << std::endl;
		return abs(rhs-lhs) <= epsilon; // "<=", should also work for zero
	}
};

template <typename T>
struct RoughlyEqualToScalar<T, true>
{
	static bool roughly_equal_to (const T &lhs, const T &rhs)
	{
		assert(std::numeric_limits<T>::is_integer);
		return lhs == rhs;
	}
};


template <typename T, bool is_arithmetic>
struct RoughlyEqualTo
{
	static bool roughly_equal_to (const T &lhs, const T &rhs)
	{ return detail::RoughlyEqualToScalar<T, std::numeric_limits<T>::is_integer>::roughly_equal_to(lhs, rhs); }
};

template <typename T>
struct RoughlyEqualTo<T, false>
{
	static bool roughly_equal_to (const T &lhs, const T &rhs)
	{
		typedef typename T::value_type V;
		assert(lhs.size() == rhs.size());
#ifdef __GXX_EXPERIMENTAL_CXX0X__
		for (auto li = lhs.begin(), ri = rhs.begin(); li != lhs.end(); ++li, ++ri)
#else
		typedef typename T::const_iterator TCI;
		for (TCI li = lhs.begin(), ri = rhs.begin(); li != lhs.end(); ++li, ++ri)
#endif
		{
			assert(ri != rhs.end());
			::cvmlcpp::roughly_equal_to<V> equal_to_V;
			if (!equal_to_V(*li, *ri))
				return false;
		}

		return true;
	}
};

} // end namespace detail

template <typename T>
struct roughly_equal_to : public std::binary_function<T, T, bool>
{
	bool operator() (const T &lhs, const T &rhs) const
#ifdef __GXX_EXPERIMENTAL_CXX0X__
	{ return detail::RoughlyEqualTo<T, std::is_arithmetic<T>::value>::roughly_equal_to(lhs, rhs); }
#else
	{ return detail::RoughlyEqualTo<T, std::is_arithmetic<T>::value>::roughly_equal_to(lhs, rhs); }
#endif
};

} // end namespace cvmlcpp

/*
 * Now that the Functors required by Math are defined, include it :-)
 */
#include <cvmlcpp/base/Enums>
#include <cvmlcpp/math/Math>
#include <cvmlcpp/base/Meta>

namespace cvmlcpp
{


template <typename T>
class Printer : public std::unary_function<const T, void>
{
	public:
		typedef T value_type;
		typedef void result_type;

		Printer(std::string _delimiter = " ") : delimiter(_delimiter)
		{ }

		void operator()(const T &item) const
		{
			std::cout << item << delimiter;
		}

	private:
		const std::string delimiter;
};

/*
 * Functors for std::pair
 */

// Select "first" from a pair<>
template <typename T, typename U>
struct Select1st : public std::unary_function<const std::pair<T, U>, T>
{
	typedef std::pair<T, U> value_type;
	typedef T result_type;
	T operator()(const std::pair<T, U> &p) const { return p.first; }
};

// Select "second" from a pair<>
template <typename T, typename U>
struct Select2nd : public std::unary_function<const std::pair<T, U>, U>
{
	typedef std::pair<T, U> value_type;
	typedef U result_type;
	U operator()(const std::pair<T, U> &p) const { return p.second; }
};

/*
 * Variants on the "bool ? trueValue : falseValue" construction
 */

// For input that is itself boolean
template <typename T>
class Chooser1 : public std::unary_function<bool, T>
{
	public:
		typedef T value_type;
		typedef T result_type;

		Chooser1(T trueValue, T falseValue) :
				_t(trueValue), _f(falseValue) { }

		T operator()(const bool &o) const
		{
			return o ? _t : _f;
		}

	private:
		const T _t, _f;
};

// For unary functors returning a boolean
template <typename T, typename R, class Operator>
class Chooser1op : public std::unary_function<T, R>
{
	public:
		typedef T value_type;
		typedef T result_type;

		Chooser1op(const Operator op,
			   const R trueValue,
			   const R falseValue) :
				_op(op), _t(trueValue), _f(falseValue) { }

		R operator()(const T &o) { return _op(o) ? _t : _f; }

	private:
		Operator _op;
		const R _t, _f;
};

// For binary functors returning a boolean
template <typename T, typename R, class Operator>
class Chooser2op : public std::binary_function<T, T, R>
{
	public:
		typedef T value_type;
		typedef T result_type;

		Chooser2op(const Operator op,
			   const R trueValue,
			   const R falseValue) :
				_op(op), _t(trueValue), _f(falseValue) { }

		R operator()(const T &lhs, const T &rhs)
		{ return _op(lhs, rhs) ? _t : _f; }

	private:
		Operator _op;
		const R _t, _f;
};

/**
 * Functors for container-like Vectors
 */

template <typename Vector, typename register_type>
class Rotator3D
{
	public:
		typedef typename Vector::value_type value_type;
		typedef Vector result_type;

		Rotator3D(std::size_t axis, register_type angle) : _axis(axis),
			cosA(std::cos(angle)), sinA(std::sin(angle))
		{ }

		const Vector operator()(const Vector &v) const
		{
			assert(v.size() == 3u);
			Vector rv(v);

			switch (_axis)
			{
				case X: rv[Y] = v[Y] * cosA - v[Z] * sinA;
					rv[Z] = v[Y] * sinA + v[Z] * cosA;
					break;
				case Y: rv[X] = v[Z] * sinA + v[X] * cosA;
					rv[Z] = v[Z] * cosA - v[X] * sinA;
					break;
				case Z: rv[X] = v[X] * cosA - v[Y] * sinA;
					rv[Y] = v[X] * sinA + v[Y] * cosA;
					break;
				default: assert(false);
			}

			return rv;
		}

	private:
		const std::size_t _axis;
		const register_type cosA, sinA;
};

template <typename T>
class LinearFunc
{
	public:
		typedef T value_type;
		typedef T result_type;

		LinearFunc(const T a, const T b) : _a(a), _b(b) {}

		T operator()(const T &x) const { return _a*x + _b; }

	private:
		const T _a, _b;
};

// A quadratic function: y = a*x^2 + b*x + c
template <typename T>
class QuadraticFunc
{
	public:
		typedef T value_type;
		typedef T result_type;

		QuadraticFunc(const T a, const T b, const T c) :
			_a(a), _b(b), _c(c) {}

		T operator()(const T &x) const { return _a*x*x + _b*x + _c; }

	private:
		const T _a, _b, _c;
};

template <typename T>
struct Square
{
	typedef T value_type;
	typedef T result_type;

	T operator()(const T &x) const { return x*x; }
};

// Gaussian Normal
template <typename T>
class Gaussian
{
	public:
		typedef T value_type;
		typedef T result_type;

		Gaussian(const T mu, const T sigmaSq):_mu(mu),
			_OneOver2SigmaSq(T(1) / (T(2) * sigmaSq)),
 			_OneOverSigma(T(1) / std::sqrt(sigmaSq)) { }

		T operator()(const T &x) const
		{
			const T OneOverSqrt2Pi = T(1) /
				std::sqrt( T(2)*Constants<T>::pi() );
			const T xmu = x - _mu;

			return OneOverSqrt2Pi * _OneOverSigma *
				std::exp(- xmu*xmu * _OneOver2SigmaSq );
		}

	private:
		const T _mu, _OneOver2SigmaSq, _OneOverSigma;
};

template <typename T>
class Clamp
{
	public:
		typedef T value_type;
		typedef T result_type;

		Clamp(const T low, const T high) : _low(low), _high(high) { }

		T operator()(const T &x) const
		{ return std::min(_high, std::max(x, _low)); }

	private:
		const T _low, _high;
};

namespace detail
{

template <bool value_is_arithmetic, bool result_is_arithmetic>
struct Derivative_
{
	template <class Function>
	static typename Function::value_type derivative(
			const Function &f,
			const typename Function::value_type &x)
	{
		typedef typename Function::value_type   value_type;
		typedef typename ValueType<value_type>::value_type T;

		// Find minimum value for h
		const std::size_t digits = std::numeric_limits<T>::digits;
		const std::size_t bits   = CHAR_BIT * sizeof(std::size_t);
		const std::size_t N      = std::min(digits, bits-1);

		assert(1ul << N);
		using std::abs;
		const value_type h = (x != 0) ? abs(x) / T(1ul << (1ul+N/2)) :
					T(1) / T(1ul << (1ul+N/2));
		assert(h > 0);
		return (f(x+h) - f(x-h)) / (h+h);
	}
};

template <>
struct Derivative_<false, true>
{
	template <class Function>
	static typename Function::value_type derivative(
			const Function &f,
			const typename Function::value_type &x)
	{
		typedef typename Function::value_type   value_type;
		typedef typename ValueType<value_type>::value_type T;

		// Find minimum value for h
		const std::size_t digits = std::numeric_limits<T>::digits;
		const std::size_t bits   = CHAR_BIT * sizeof(std::size_t);
		const std::size_t N      = std::min(digits, bits-1);

		assert(1ul << N);

		typename Function::value_type x_min_h, x_plus_h, deriv;
		for (std::size_t i = 0; i < x.size(); ++i)
		{
			using std::abs;
			const T h = (x[i] != 0) ? abs(x[i]) / T(1ul << (1ul+N/2)) :
						T(1) / T(1ul << (1ul+N/2));
			assert(h > 0);
			x_min_h  = x; x_min_h[i]  -= h;
			x_plus_h = x; x_plus_h[i] += h;
			deriv[i] = (f(x_plus_h) - f(x_min_h)) / (h+h);
		}
		return deriv;
	}
};

} // end namespace detail

template <class Function>
class Derivative
{
	public:
		typedef typename Function:: value_type  value_type;
		typedef typename Function::result_type result_type;

		Derivative(const Function &f) : f_(f) { }

		value_type operator()(const value_type &x) const
		{
			return detail::Derivative_<  std::is_arithmetic< value_type>::value,
				std::is_arithmetic<result_type>::value >::derivative(f_, x);
		}

	private:
		const Function f_;

};

/*
 * Functors that operate on containers. Should remain undocumented until it is
 * clear that this does not duplicate functionality offered by BOOST.
 */

template <typename Container>
struct ContainerSorter
{
	ContainerSorter () {}

	void operator()(Container &container) const
	{
		std::sort(container.begin(), container.end());
	}
};

template <class Container, class Operator>
class UnaryOperateInserter
{
	public:
		UnaryOperateInserter(Container &container, Operator op) :
			_container(container), _op(op) { }

		template <typename T>
		void operator()(const T &object)
		{ _container.insert(_op(object)); }

	private:
		Container &_container;
		Operator _op;
};

template <class Container, class Operator>
class BinaryOperateInserter
{
	public:
		BinaryOperateInserter(Container &container, Operator op) :
			_container(container), _op(op) { }

		template <typename T, typename U>
		void operator()(const T &lhs, const T &rhs)
		{ _container.insert(_op(lhs, rhs)); }

	private:
		Container &_container;
		Operator _op;
};

template <class Map, class Operator>
class MapPairInserter
{
	public:
		MapPairInserter(Map &mp) : _mp(mp) { }

		template <typename T, typename U>
		void operator()(const std::pair<T, U> &pr) const
		{ _mp[pr.first] = pr.second; }

	private:
		Map &_mp;
};

template <class Map, class Operator>
class MapPairOperateInserter
{
	public:
		MapPairOperateInserter(Map &mp, Operator op) :
			_mp(mp), _op(op) { }

		template <typename T, typename U>
		void operator()(const std::pair<T, U> &pr)
		{ _mp[pr.first] = _op(pr.second); }

	private:
		Map &_mp;
		Operator _op;
};

/*
template <typename T, typename U, class Operator>
class PairFirstBinaryOperator : public std::binary_function<T, T, bool>
{
	public:
		PairFirstBinaryOperator() {}

		bool operator()(const std::pair<T, U> &lhs,
				const std::pair<T, U> &rhs)
		{ return _op(lhs.first, rhs.first); }

	private:
		Operator _op;
};
*/

template <typename T, typename U, class Operator = std::less<T> >
class PairFirstCompare : public std::binary_function<T, T, bool>
{
	public:
		typedef T value_type;
		typedef bool result_type;

		PairFirstCompare() {}

		bool operator()(const T &lhs, const std::pair<T, U> &rhs) const
		{ return _op(lhs, rhs.first); }

		bool operator()(const std::pair<T, U> &lhs, const T &rhs) const
		{ return _op(lhs.first, rhs); }

		bool operator()(const std::pair<T, U> &lhs,
				const std::pair<T, U> &rhs) const
		{ return _op(lhs.first, rhs.first); }

	private:
		Operator _op;
};


/*
 * Nesting allows a kind of preprocessing.
 */
/*
template <class Inner, class Outer, typename T, typename U>
class Nest1 : public std::unary_function<T, U>
{
	public:
		Nest1(Inner &inner, Outer &outer) :
			_inner(inner), _outer(outer) { }

		U operator() (const T &o)
		{ return _outer(_inner(o)); }

	private:
		Inner	_inner;
		Outer	_outer;
};

template <class Inner, class Outer, typename T, typename U>
class Nest2 : public std::binary_function<T, T, U>
{
	public:
		Nest2(Inner &inner, Outer &outer) :
			_inner(inner), _outer(outer) { }

		U operator() (const T &lhs)
		{ return _outer(_inner(lhs), _inner(rhs)); }

	private:
		Inner	_inner;
		Outer	_outer;
};
*/

} // namespace
