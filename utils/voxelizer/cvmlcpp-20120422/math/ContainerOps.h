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

#ifndef CVMLCPP_CONTAINER_OPS
#define CVMLCPP_CONTAINER_OPS 1

#include <cmath>

#include <type_traits>

#include <cvmlcpp/base/Enums>
#include <cvmlcpp/base/Meta>

#include <cvmlcpp/base/use_omp.h>
#include <omptl/omptl_algorithm>

namespace cvmlcpp
{

namespace detail
{

/*
 * Vector Ops "+=" "-=" etc.
 */

// Vector-Scalar Ops
template <class Container, class Op, bool is_arithmetic>
struct _doOpIs
{
	template <class RHS>
	static const Container &execute(Container &lhs, const RHS &rhs)
	{
		std::transform(lhs.begin(), lhs.end(), lhs.begin(),
				std::bind2nd(Op(), rhs));
		return lhs;
	}
};

// Vector-Vector ops
template <class Container, class Op>
struct _doOpIs<Container, Op, false>
{
	template <class RHS>
	static const Container &execute(Container &lhs, const RHS &rhs)
	{
		std::transform(lhs.begin(), lhs.end(), rhs.begin(),
				lhs.begin(), Op());
		return lhs;
	}
};

template <class Container, class Op>
struct doOpIs
{
	template <class RHS>
	static const Container &execute(Container &lhs, const RHS &rhs)
	{
		return _doOpIs<Container, Op,
				std::is_arithmetic<RHS>::value>::
					execute(lhs, rhs);
	}
};

template <class Container, class Op, bool is_arithmetic>
struct _doOp
{
	template <class RHS>
	static const Container execute(const Container &lhs, const RHS &rhs)
	{
		Container result = lhs;
		std::transform(result.begin(), result.end(),
				result.begin(), std::bind2nd(Op(), rhs));

		return result;
	}

	template <class RHS>
	static const Container execute(const RHS &lhs, const Container &rhs)
	{
		Container result = rhs;
		std::transform(result.begin(), result.end(),
				result.begin(), std::bind1st(Op(), lhs));

		return result;
	}
};

template <class Container, class Op>
struct _doOp<Container, Op, false>
{
	template <class RHS>
	static const Container execute(const Container &lhs, const RHS &rhs)
	{
		assert(lhs.size() == rhs.size());
		Container result = lhs;
		std::transform(result.begin(), result.end(),
				rhs.begin(), result.begin(), Op());

		return result;
	}

	template <class LHS>
	static const Container execute(const LHS &lhs, const Container &rhs)
	{
		Container result = rhs;
		std::transform(lhs.begin(), lhs.end(),
				result.begin(), result.begin(), Op());

		return result;
	}

	static const Container execute(const Container &lhs,
					const Container &rhs)
	{
		Container result = rhs;
		std::transform(lhs.begin(), lhs.end(),
				result.begin(), result.begin(), Op());

		return result;
	}
};

/*
 * Vector Ops "+" "-" etc.
 */

template <class Container, class Op>
struct doOp
{
	template <class RHS>
	static const Container execute(const Container &lhs, const RHS &rhs)
	{
		return _doOp<Container, Op,
				std::is_arithmetic<RHS>::value>::
					execute(lhs, rhs);
	}

	template <class LHS>
	static const Container execute(const LHS &lhs, const Container &rhs)
	{
		return _doOp<Container, Op,
				std::is_arithmetic<LHS>::value>::
					execute(lhs, rhs);
	}

	static const Container execute(const Container &lhs,
					const Container &rhs)
	{
		return _doOp<Container, Op, false>::execute(lhs, rhs);
	}

};

/*
 * Matrix-Vector Ops
 */

template <template <typename Tm, std::size_t Dm, typename A> class Array_t,
	  typename Ta, typename Aux,
	  class Vector_t, class RVector_t>
void mat_vec_mult(const Array_t<Ta, 2, Aux> &A, const Vector_t &v, RVector_t &r)
{
	typedef cvmlcpp::array_traits<Array_t, Ta, 2, Aux> ATraits;

	const std::size_t N = r.size();

	// Dimensions must match
	assert(N == ATraits::shape(A)[0]);

#ifdef _OPENMP
	#pragma omp parallel for
#endif
	for (int i = 0; i < int(N); ++i)
	{
		assert(std::size_t(std::distance(A[i].begin(),A[i].end())) == v.size());
 		r[i] = std::inner_product(A[i].begin(), A[i].end(),
 					  v.begin(), 0.0);
	}
}

/*
 * Matrix-Matrix ops
 */

template <template<typename T, std::size_t N, typename Aux> class ArrayLhs,
	  template<typename T, std::size_t N, typename Aux> class ArrayRhs,
	  template<typename T, std::size_t N, typename Aux> class ArrayRes,
	  typename Tlhs, typename Trhs, typename Tres,
	  typename AuxLhs, typename AuxRhs, typename AuxRes>
void mat_mat_mult(const ArrayLhs<Tlhs, 2, AuxLhs> &lhs,
		  const ArrayRhs<Trhs, 2, AuxRhs> &rhs,
		  	ArrayRes<Tres, 2, AuxRes> &res)
{
	typedef cvmlcpp::array_traits<ArrayLhs, Tlhs, 2, AuxLhs>	LhsTraits;
	typedef cvmlcpp::array_traits<ArrayRhs, Trhs, 2, AuxRhs>	RhsTraits;
	typedef cvmlcpp::array_traits<ArrayRes, Tres, 2, AuxRes>	ResTraits;

	// Matrix dimensions must match
	assert(LhsTraits::shape(lhs)[Y] == RhsTraits::shape(rhs)[X]);
	const std::size_t N = LhsTraits::shape(lhs)[Y];

	const std::size_t dims [] = { LhsTraits::shape(lhs)[X],
			      RhsTraits::shape(rhs)[Y] };
	ResTraits::resize(res, dims);

	const std::size_t Nx = dims[X];
	const std::size_t Ny = dims[Y];

#ifdef _OPENMP
	#pragma omp parallel for
#endif
	for (int i = 0; i < int(Nx); ++i)
	for (std::size_t j = 0; j < Ny; ++j)
	{
		Tres sum = 0.0;
		for (std::size_t k = 0; k < N; ++k)
			sum += lhs[i][k] * rhs[k][j];

		res[i][j] = sum;
	}
}

template <class Operator>
struct _MatMatOp
{

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename AuxLhs, typename AuxRhs, std::size_t N>
static Array_t<Tm, N, AuxLhs> matMatOp( const Array_t<Tm, N, AuxLhs> &lhs,
					const Array_t<Tm, N, AuxRhs> &rhs)
{
	typedef cvmlcpp::array_traits<Array_t, Tm, N, AuxLhs> ATLHS;
	typedef cvmlcpp::array_traits<Array_t, Tm, N, AuxRhs> ATRHS;

	assert(std::equal(ATLHS::shape(lhs), ATLHS::shape(lhs)+N,
			  ATRHS::shape(rhs)));

	Array_t<Tm, N, AuxLhs> result = ATLHS::create(ATLHS::shape(lhs));
	omptl::transform(ATLHS::begin(lhs), ATLHS::end(lhs), ATRHS::begin(rhs),
			 ATLHS::begin(result), Operator());

	return result;
}

};

/*
 * Matrix-Scalar ops.
 */

template <class Operator>
struct _MatOp
{

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename Tv, typename A, std::size_t N>
static Array_t<Tm, N, A> doOp(const Array_t<Tm, N, A> &m, const Tv v)
{
	typedef cvmlcpp::array_traits<Array_t, Tm, N, A> AT;

	Array_t<Tm, N, A> result = AT::create(AT::shape(m));

	omptl::transform(m.begin(), m.end(), result.begin(),
			 std::bind2nd(Operator(), v));

	return result;
}

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename Tv, typename A, std::size_t N>
static Array_t<Tm, N, A> doOp(const Tv v, const Array_t<Tm, N, A> &m)
{
	typedef cvmlcpp::array_traits<Array_t, Tm, N, A> AT;

	Array_t<Tm, N, A> result = AT::create(AT::shape(m));

	omptl::transform(m.begin(), m.end(), result.begin(),
			 std::bind1st(Operator(), v));

	return result;
}

};

} // End Namespace detail

} // End Namespace cvmlcpp

/*
 * Matrix-Matrix Ops
 */
template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename AuxLhs, typename AuxRhs>
Array_t<Tm, 2, AuxLhs> operator*(const Array_t<Tm, 2, AuxLhs> &lhs,
				 const Array_t<Tm, 2, AuxRhs> &rhs)
{
	Array_t<Tm, 2, AuxLhs> result;
	cvmlcpp::detail::mat_mat_mult(lhs, rhs, result);
	return result;
}

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename AuxLhs, typename AuxRhs, std::size_t N>
Array_t<Tm, N, AuxLhs> operator+(const Array_t<Tm, N, AuxLhs> &lhs,
				 const Array_t<Tm, N, AuxRhs> &rhs)
{
	return cvmlcpp::detail::_MatMatOp<std::plus<Tm> >::matMatOp(lhs, rhs);
}

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename AuxLhs, typename AuxRhs, std::size_t N>
Array_t<Tm, N, AuxLhs> operator-(const Array_t<Tm, N, AuxLhs> &lhs,
				 const Array_t<Tm, N, AuxRhs> &rhs)
{
	return cvmlcpp::detail::_MatMatOp<std::minus<Tm> >::matMatOp(lhs, rhs);
}

/*
 * Matrix-Vector Ops
 */

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename Tv, typename A, std::size_t N>
cvmlcpp::DynamicVector<Tv> operator*(const Array_t<Tm, 2, A> &m, const cvmlcpp::StaticVector<Tv, N> &v)
{
	using namespace cvmlcpp;
	typedef array_traits<Array_t, Tm, 2, A>ArrayTraits;

	// Users: the dimensionality of the matrix must equal the vector length
	assert(ArrayTraits::shape(m)[1] == N);

	DynamicVector<Tv> result( (ArrayTraits::shape(m)[0]) );
	detail::mat_vec_mult(m, v, result);
	return result;
}

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename Tv, typename A>
cvmlcpp::DynamicVector<Tv> operator*(const Array_t<Tm, 2, A> &m, const cvmlcpp::DynamicVector<Tv> &v)
{
	using namespace cvmlcpp;
	typedef cvmlcpp::array_traits<Array_t, Tm, 2, A>ArrayTraits;

	// Users: the dimensionality of the matrix must equal the vector length
	assert(ArrayTraits::shape(m)[1] == v.size());

	DynamicVector<Tv> result( (ArrayTraits::shape(m)[0]) );
	detail::mat_vec_mult(m, v, result);
	return result;
}

/*
 * Matrix-Scalar Ops
 */
template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename Tv, typename A, std::size_t N>
Array_t<Tm, N, A> operator+(const Array_t<Tm, N, A> &m, const Tv v)
{
	return cvmlcpp::detail::_MatOp<std::plus<Tm> >::doOp(m, v);
}

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename Tv, typename A, std::size_t N>
Array_t<Tm, N, A> operator-(const Array_t<Tm, N, A> &m, const Tv v)
{
	return cvmlcpp::detail::_MatOp<std::minus<Tm> >::doOp(m, v);
}

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename Tv, typename A, std::size_t N>
Array_t<Tm, N, A> operator*(const Array_t<Tm, N, A> &m, const Tv v)
{
	return cvmlcpp::detail::_MatOp<std::multiplies<Tm> >::doOp(m, v);
}

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename Tv, typename A, std::size_t N>
Array_t<Tm, N, A> operator/(const Array_t<Tm, N, A> &m, const Tv v)
{
	return cvmlcpp::detail::_MatOp<std::divides<Tm> >::doOp(m, v);
}

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename Tv, typename A, std::size_t N>
Array_t<Tm, N, A> operator+(const Tv v, const Array_t<Tm, N, A> &m)
{
	return cvmlcpp::detail::_MatOp<std::plus<Tm> >::doOp(v, m);
}

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename Tv, typename A, std::size_t N>
Array_t<Tm, N, A> operator-(const Tv v, const Array_t<Tm, N, A> &m)
{
	return cvmlcpp::detail::_MatOp<std::minus<Tm> >::doOp(v, m);
}

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename Tv, typename A, std::size_t N>
Array_t<Tm, N, A> operator*(const Tv v, const Array_t<Tm, N, A> &m)
{
	return cvmlcpp::detail::_MatOp<std::multiplies<Tm> >::doOp(v, m);
}

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename Tv, typename A, std::size_t N>
Array_t<Tm, N, A> operator/(const Tv v, const Array_t<Tm, N, A> &m)
{
	return cvmlcpp::detail::_MatOp<std::divides<Tm> >::doOp(v, m);
}

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename Tv, typename A, std::size_t N>
Array_t<Tm, N, A> &operator+=(Array_t<Tm, N, A> &m, const Tv v)
{
	m = m + v;
	return m;
}

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename Tv, typename A, std::size_t N>
Array_t<Tm, N, A> &operator-=(Array_t<Tm, N, A> &m, const Tv v)
{
	m = m - v;
	return m;
}

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename Tv, typename A, std::size_t N>
Array_t<Tm, N, A> &operator*=(Array_t<Tm, N, A> &m, const Tv v)
{
	m = m * v;
	return m;
}

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename Tv, typename A, std::size_t N>
Array_t<Tm, N, A> &operator/=(Array_t<Tm, N, A> &m, const Tv v)
{
	m = m / v;
	return m;
}


/*
 * Matrix Ops "+=" "-=" etc.
 */

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename AuxLhs, typename AuxRhs, std::size_t N>
void operator+=(Array_t<Tm, N, AuxLhs> &lhs, const Array_t<Tm, N, AuxRhs> &rhs)
{
	typedef cvmlcpp::array_traits<Array_t, Tm, N, AuxLhs>	LHSTraits;
	typedef cvmlcpp::array_traits<Array_t, Tm, N, AuxRhs>	RHSTraits;

	assert(equal(LHSTraits::shape(lhs), LHSTraits::shape(lhs)+N,
		     RHSTraits::shape(rhs)));

	omptl::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(),
			 std::plus<Tm>());
}

template <template<typename T, std::size_t N, typename Aux> class Array_t,
	  typename Tm, typename AuxLhs, typename AuxRhs, std::size_t N>
void operator-=(Array_t<Tm, N, AuxLhs> &lhs, const Array_t<Tm, N, AuxRhs> &rhs)
{
	typedef cvmlcpp::array_traits<Array_t, Tm, N, AuxLhs>	LHSTraits;
	typedef cvmlcpp::array_traits<Array_t, Tm, N, AuxRhs>	RHSTraits;

	assert(equal(LHSTraits::shape(lhs), LHSTraits::shape(lhs)+N,
		     RHSTraits::shape(rhs)));

	omptl::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(),
			 std::minus<Tm>());
}

/*
 * Operations on containers
 */
namespace cvmlcpp {


#define CVMLCPP_UNARY_FUNCTOR(F) \
namespace detail { template <typename T> struct unary_fun_##F { T operator()(const T x) const { return std::F(x); } }; }

#define CVMLCPP_STATIC_VECTOR_UNARY_FUNC(F) \
template <template <typename TT, std::size_t DD> class Vect, typename T, std::size_t D>\
Vect<T, D> F(const Vect<T, D> &x) \
{ \
	Vect<T, D> y; \
	omptl::transform(x.begin(), x.end(), y.begin(), /*cvmlcpp::*/detail::unary_fun_##F<T>() ); \
	return y; \
} \
template <template <typename TT> class Vect, typename T>\
Vect<T> F(const Vect<T> &x) \
{ \
	Vect<T> y; \
	omptl::transform(x.begin(), x.end(), y.begin(), /*cvmlcpp::*/detail::unary_fun_##F<T>() ); \
	return y; \
} \

#define CVMLCPP_CONTAINER_OPS_UNARY_FUNC(F) CVMLCPP_UNARY_FUNCTOR(F) CVMLCPP_STATIC_VECTOR_UNARY_FUNC(F)

CVMLCPP_CONTAINER_OPS_UNARY_FUNC(abs)
CVMLCPP_CONTAINER_OPS_UNARY_FUNC(sin)
CVMLCPP_CONTAINER_OPS_UNARY_FUNC(cos)
CVMLCPP_CONTAINER_OPS_UNARY_FUNC(tan)
CVMLCPP_CONTAINER_OPS_UNARY_FUNC(asin)
CVMLCPP_CONTAINER_OPS_UNARY_FUNC(acos)
CVMLCPP_CONTAINER_OPS_UNARY_FUNC(atan)
CVMLCPP_CONTAINER_OPS_UNARY_FUNC(norm)
CVMLCPP_CONTAINER_OPS_UNARY_FUNC(ceil)
CVMLCPP_CONTAINER_OPS_UNARY_FUNC(cosh)
CVMLCPP_CONTAINER_OPS_UNARY_FUNC(sinh)
CVMLCPP_CONTAINER_OPS_UNARY_FUNC(tanh)
CVMLCPP_CONTAINER_OPS_UNARY_FUNC(floor)
CVMLCPP_CONTAINER_OPS_UNARY_FUNC(log)
CVMLCPP_CONTAINER_OPS_UNARY_FUNC(log10)
CVMLCPP_CONTAINER_OPS_UNARY_FUNC(sqrt)
//CVMLCPP_CONTAINER_OPS_UNARY_FUNC()

#define CVMLCPP_BINARY_FUNCTOR(F) \
/*namespace cvmlcpp { */namespace detail { template <typename T> struct binary_fun_##F { T operator()(const T x, const T y) const { return std::F(x, y); } }; }

#define CVMLCPP_STATIC_VECTOR_BINARY_FUNC(F) \
template <template <typename TT, std::size_t DD> class Vect, typename T, std::size_t D>\
Vect<T, D> F(const Vect<T, D> &x, const Vect<T, D> &y) \
{ \
	Vect<T, D> z; \
	omptl::transform(x.begin(), x.end(), y.begin(), z.begin(), /*cvmlcpp::*/detail::binary_fun_##F<T>() ); \
	return z; \
} \
template <template <typename TT> class Vect, typename T>\
Vect<T> F(const Vect<T> &x, const Vect<T> &y) \
{ \
	Vect<T> z; \
	omptl::transform(x.begin(), x.end(), y.begin(), z.begin(), /*cvmlcpp::*/detail::binary_fun_##F<T>() ); \
	return z; \
}

#define CVMLCPP_CONTAINER_OPS_BINARY_FUNC(F) CVMLCPP_BINARY_FUNCTOR(F) CVMLCPP_STATIC_VECTOR_BINARY_FUNC(F)

CVMLCPP_CONTAINER_OPS_BINARY_FUNC(min)
CVMLCPP_CONTAINER_OPS_BINARY_FUNC(max)
CVMLCPP_CONTAINER_OPS_BINARY_FUNC(atan2)
CVMLCPP_CONTAINER_OPS_BINARY_FUNC(exp)
//CVMLCPP_CONTAINER_OPS_BINARY_FUNC(mod)
CVMLCPP_CONTAINER_OPS_BINARY_FUNC(pow)
//CVMLCPP_CONTAINER_OPS_BINARY_FUNC()


#define CVMLCPP_CONTAINER_OPS_ARG_FUNC(F) \
template <template <typename TT, std::size_t DD> class Vect, typename T, std::size_t D, typename U>\
Vect<T, D> F(const Vect<T, D> &x, const U &arg) \
{ \
	Vect<T, D> y; \
	omptl::transform(x.begin(), x.end(), y.begin(), \
			std::bind2nd(/*cvmlcpp::*/detail::binary_fun_##F<T>(), arg)); \
	return y; \
} \
template <template <typename TT> class Vect, typename T, typename U>\
Vect<T> F(const Vect<T> &x, const U &arg) \
{ \
	Vect<T> y; \
	omptl::transform(x.begin(), x.end(), y.begin(), \
			std::bind2nd(/*cvmlcpp::*/detail::binary_fun_##F<T>(), arg)); \
	return y; \
}


CVMLCPP_CONTAINER_OPS_ARG_FUNC(exp)
//CVMLCPP_CONTAINER_OPS_ARG_FUNC(mod)
CVMLCPP_CONTAINER_OPS_ARG_FUNC(pow)
//CVMLCPP_CONTAINER_OPS_ARG_FUNC()

} // end namespace

#endif
