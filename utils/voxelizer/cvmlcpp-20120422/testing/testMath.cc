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

#include <cassert>
#include <iostream>

#include <cvmlcpp/base/Functors>
#include <cvmlcpp/base/IDGenerators>
#include <cvmlcpp/math/Math>
#include <cvmlcpp/math/Vectors>

using namespace cvmlcpp;

template <typename T>
struct SquareDeriv
{
	typedef T value_type;
	typedef T result_type;
	T operator()(const T &x) const { return T(2)*x; }
};

template <typename T>
struct Pow3
{
	typedef T value_type;
	typedef T result_type;
	T operator()(const T &x) const { return x * x * x; }
};

template <typename T>
struct SQRT
{
	typedef T value_type;
	typedef T result_type;
	T operator()(const T &x) const { return std::sqrt(std::abs(x)); }
};

template <typename T>
struct RKSquare
{
	typedef T value_type;
	typedef T result_type;
	T operator()(T t, T y) const { return t*t-y; }
};

template <class T>
void testDiagHungarian_(Matrix<T, 2> &m)
{
	assert(m.extents()[X] == m.extents()[Y]);
	std::vector<std::pair<std::size_t, std::size_t> > mates;

	cvmlcpp::find_matching(m, mates);
	assert(mates.size() == m.extents()[X]);
	for (std::size_t i = 0; i < mates.size(); ++i)
	{
		assert(mates[i].first  == i);
		assert(mates[i].second == i);
	}
}

template <class T>
void testHungarian1()
{
	using namespace cvmlcpp;

	const T data [] = { 1, 9, 9,
			    9, 2, 9,
			    9, 9, 3 };
	int dims [] = { 3,3 };
	Matrix<T, 2> m(dims);
	std::copy(data, data+9, m.begin());
	testDiagHungarian_(m);
}

template <class T>
void testHungarian2()
{
	using namespace cvmlcpp;

	const T data [] = { 1, 9, 9,
			    9, 1, 9,
			    9, 3, 4 };
	int dims [] = { 3,3 };
	Matrix<T, 2> m(dims);
	std::copy(data, data+9, m.begin());
	testDiagHungarian_(m);
}

template <typename T>
struct TwoToOne
{
	typedef StaticVector<T, 2> value_type;
	typedef T result_type;

	result_type operator()(const value_type &x) const
	{ return (std::pow(x[0]-T(3), T(2))+1) + (std::pow(x[1]-T(2), T(2))+1); }
};

template <typename T>
struct TwoToOneDeriv
{
	typedef StaticVector<T, 2> value_type;
	typedef T result_type;

	value_type operator()(const value_type &x) const
	{
		value_type deriv;
		deriv[0] = (2*x[0] - 6);
		deriv[1] = (2*x[1] - 4);
//std::cout << "x: " << x  << " d: " << deriv << std::endl;
		return deriv;
	}
};

template <typename T>
void testOptimize()
{
	// -(x-2)^2, = -x^2 + 4x - 4 : maximum at 2
	const QuadraticFunc<T> maximize = QuadraticFunc<T>(-1, 4, -4);
	const LinearFunc<T> dmax = LinearFunc<T>(-2, 4);
	//std::cout << "Max + deriv: " << optimize( maximize, dmax, T(-16), T(16) ) << std::endl;
	assert(optimize( maximize, dmax, T(-16), T(16) ) == 2);
	//std::cout << "Max: " << optimize( maximize, T(-16), T(16) ) << std::endl;
	assert(optimize( maximize, T(-16), T(16) ) == 2);

	StaticVector<T, 2> a, b, c, l, h;
	a[0]=    a[1] = - 1;
	b[0]= 6; b[1] =   4;
	c[0]=-9; c[1] = - 4;
	l[0]=    l[1] = -16;
	h[0]=    h[1] =  16;

	StaticVector<T, 2> a_, b_;
	a_[0]=    a_[1] = - 2;
	b_[0]= 6; b_[1] =   4;

	const QuadraticFunc< StaticVector<T, 2> > mx = QuadraticFunc< StaticVector<T, 2> >(a, b, c);
	const LinearFunc< StaticVector<T, 2> > mx_ = LinearFunc< StaticVector<T, 2> >(a_, b_);
	StaticVector<T, 2> x; x[0] = 3; x[1] = 2;
	assert( optimize( mx, mx_, l, h) == x );
	//std::cout << optimize( mx, mx_, l, h).to_string() << std::endl;

	//StaticVector<T, 2> x = optimize( mx, l, h);
	//std::cout << optimize(mx, l, h).to_string() << std::endl;

	assert( optimize(mx, l, h) == x);


	//std::cout << optimize(TwoToOne<T>(), TwoToOneDeriv<T>(), l, h) << std::endl;
	//std::cout << optimize(TwoToOne<T>(), l, h) << std::endl;
	assert( abs(optimize(TwoToOne<T>(), l, h) - x) < 0.0001);
	assert( abs(optimize(TwoToOne<T>(), TwoToOneDeriv<T>(), l, h) - x) < 0.0001);
}

int main()
{
	assert(gcd(5, 7) == 1);
	assert(gcd(5., 7.) == 1.);
	assert(binomial(7, 5) == 21);

	for (std::size_t i = 0; i < CHAR_BIT*sizeof(std::size_t); ++i)
	{
		//~ std::cout << i << " " << (std::size_t(1) << i) << " " << log2(std::size_t(1) << i) << " " << log2(1 + (std::size_t(1) << i)) << std::endl;
		assert(log2(std::size_t(1) << i) == i);
	}

	// result should be largest smaller-equal than log2(i)
	for (std::size_t i = 1; i < CHAR_BIT*sizeof(std::size_t); ++i)
		assert(log2(1 + (std::size_t(1) << i)) == i);

	const double e = 1e-15;

	float r1 = -10.0f;
	const bool nr1 = doNewtonRaphson(Square<float>(), r1, 1000);
	assert(nr1);
// 	std::cout << r1 << std::endl;
	assert(std::abs(r1) < e);

	double r2 = -10.0;
	const bool nr2 = doNewtonRaphson(Square<double>(),
					SquareDeriv<double>(), r2);
	assert(nr2);
// 	std::cout << r2 << std::endl;
	assert(std::abs(r2) < e);

	StaticVector<double, 2u> r3 = -10.0;
	const bool nr3 = doNewtonRaphson(
			Square<StaticVector<double, 2u> >(), r3);
	assert(nr3);
// 	std::cout << r3.to_string() << std::endl;
	assert(modulus(r3) < e);

	double r4 = -10.0;
	const bool nr4 = doNewtonRaphson(Square<double>(),
				SquareDeriv<double>(), r4, -1.0, 7.0);
	assert(nr4);
// 	std::cout << r4 << std::endl;
	assert(std::abs(r4) < e);

	double r5 = -10.0;
	const bool nr5 = doNewtonRaphson(Pow3<double>(), r5, -1.0, 7.0);
	assert(nr5);
// 	std::cout << r5 << std::endl;
	assert(std::abs(r5) < e);

	double r6 = -10.0;
	const bool nr6 = doNewtonRaphson(SQRT<double>(), r6, -1.0, 7.0);
	assert(nr6);
// 	std::cout << r6 << std::endl;
	assert(std::abs(r6) < e);

	// Runge-Kutta
	RKSquare<double> f;
	std::vector<double> result;
	const unsigned N = 10u;
	double t0 = 0.0;
	double tN = 1.0;
// 	double dx = (tN - t0) / double(N);

	doRungeKutta(f, t0, tN, N, 0.0, result);
/*
	for (unsigned iPop = 0; iPop < result.size(); ++iPop)
	{
	    std::cout << iPop * dx << " " << result[iPop] << "\n";
	}
*/
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 2)
	for (unsigned i = 1u; i < 32u; ++i)
	{
// std::cout << i << " " << factorial(double(i)) << " " <<
// 	round(std::exp(_log_gamma(double(i)))) << std::endl;

		assert(std::log(factorial(double(i))) -
				std::tr1::tgamma(double(i+1)) < e);
	}
#endif

	// Cross product
	int _normdata [] = {1, 0, 0};
	const StaticVector<int, 3> norm(_normdata, _normdata+3);

	int _abdata [] = {0, 1, 0};
	int _acdata [] = {0, 0, 1};
	StaticVector<int, 3> ab(_abdata,_abdata+3) , ac(_acdata,_acdata+3);
	assert(norm == crossProduct(ab, ac));

	/*
	 * Statistics
	 */
	int stat_data [] = {-1, 1, -2, 2, -3, 3};
	assert(average(stat_data, stat_data+6, 0.0) == 0.0);
// 	assert(deviation(stat_data, stat_data+6, 0.0) == 2.0);
// 	assert(variance(stat_data, stat_data+6, 0.0) == 4.0);


	/*
	 * Matrices
	 */
	//
	Matrix<unsigned, 2u> m;
	identity_matrix(m, 3);
// 	for (unsigned i = 0u; i < 3u; ++i)
// 	{
// 		for (unsigned j = 0u; j < 3u; ++j)
// 	    		std::cout << m[i][j] << " ";
// 		std::cout << std::endl;
// 	}
// 	std::cout << std::endl;
	assert(std::accumulate(m.begin(), m.end(), 0) == 3);

	const unsigned d1[] = {43u, 37u};
// 	const unsigned d1[] = {4u, 3u};
	m.resize(d1);
// 	std::cout << m.extents()[X] << " x " << m.extents()[Y] << std::endl;
	assert(m.size() == d1[X] * d1[Y]);
	IncGenerator<unsigned> incGen;
	for (int i = 0; i < 100; ++i)
	{
		std::generate(m.begin(), m.end(), incGen);
// 		for (unsigned i = 0u; i < m.extents()[X]; ++i)
// 		{
// 			for (unsigned j = 0u; j < m.extents()[Y]; ++j)
// 		    		std::cout << m[i][j] << " ";
// 			std::cout << std::endl;
// 		}
// 		std::cout << std::endl;

		Matrix<unsigned, 2u> m2 = m.clone();
		transpose(m2);
// 		std::cout << m2.extents()[X] << " x " << m2.extents()[Y] << std::endl;
		assert(m2.extents()[X] == m.extents()[Y]);
		assert(m2.extents()[Y] == m.extents()[X]);
// 		for (unsigned i = 0u; i < m2.extents()[X]; ++i)
// 		{
// 			for (unsigned j = 0u; j < m2.extents()[Y]; ++j)
// 			{
// 		    		std::cout << m2[i][j] << " ";
// 				assert(m2[i][j] == m[j][i]);
// 			}
// 			std::cout << std::endl;
// 		}
// 		std::cout << std::endl;
	}

	Matrix<float, 2u> m3;
	identity_matrix(m3, 3);
	assert(invert(m3));
// 	std::cout << m.extents()[0] << " x " << m.extents()[1] << std::endl;
	for (unsigned i = 0u; i < m3.extents()[X]; ++i)
	{
		for (unsigned j = 0u; j < m3.extents()[Y]; ++j)
			assert ( m3[i][j] == ((i == j) ? 1 : 0) );
// 		std::cout << std::endl;
	}
// 	std::cout << std::endl;

	assert(m3.size() == 9u);
	const float fdata3 [] = { 4., 5., 6., 1., 3., 2., 7., 8., 9.};
	std::copy(fdata3, fdata3+m3.size(), m3.begin());
// 	for (unsigned i = 0u; i < m3.extents()[X]; ++i)
// 	{
// 		for (unsigned j = 0u; j < m3.extents()[Y]; ++j)
// 	    		std::cout << m3[i][j] << " ";
// 		std::cout << std::endl;
// 	}
// 	std::cout << std::endl;

	assert(invert(m3));
// 	for (unsigned i = 0u; i < m3.extents()[X]; ++i)
// 	{
// 		for (unsigned j = 0u; j < m3.extents()[Y]; ++j)
// 	    		std::cout << m3[i][j] << " ";
// 		std::cout << std::endl;
// 	}
// 	std::cout << std::endl;

	// Mat mult
	const float fdata4 [] = { 4, 5, 6, 1, 3, 2, 7, 8, 9, 10, 11, 12 };
	const unsigned d4[] = {4u, 3u};
	Matrix<float, 2u> m4 = Matrix<float, 2u>(d4);
	std::copy(fdata4, fdata4+m4.size(), m4.begin());

	float fdata5 [] = { 3, 5, 7, 11, 13, 17 };
	const unsigned d5[] = {3u, 2u};
	Matrix<float, 2u> m5 = Matrix<float, 2u>(d5);
	std::copy(fdata5, fdata5+m5.size(), m5.begin());

	const Matrix<float, 2u> m6 = m4 * m5;
//	mat_mat_mult(m4, m5, m6);

	const float fdata6 [] = { 125, 177, 50, 72, 194, 276, 263, 375};
// 	std::cout << "Mat mult" << std::endl;
	assert(std::equal(m6.begin(), m6.end(), fdata6));

	StaticVector<float, 2> v1 = 2.0f;
	DynamicVector<float> vr = m5 * v1;
	assert(vr.size() == 3);

	const float fdatavr [] = { 16, 36, 60 };
	assert(std::equal(vr.begin(), vr.end(), fdatavr));

	// Least Sq
	StaticVector<float, 3> y;
	y[0] = 7; y[1] = 9; y[2] = 11;

	StaticVector<float, 2> x;
	const unsigned da[] = {3u, 2u};
	Matrix<float, 2u> A(da);
	A[0][0] = 1; A[0][1] = 3;
	A[1][0] = 1; A[1][1] = 4;
	A[2][0] = 1; A[2][1] = 5;
	assert(leastSquaresFit(A, y, x));
	assert( std::abs(x[0] - 1.0) < 0.00001);
	assert( std::abs(x[1] - 2.0) < 0.00001);

// 	m4 = m5 * m6;

	m5 = m4 + 3;
	m5 = 3 - m4;
	m5 = m5 - m4;

	testHungarian1<int>();
	testHungarian1<float>();

	testHungarian2<unsigned>();
	testHungarian2<float>();

	testOptimize<double>();
	testOptimize<float>();

	return 0;
}

