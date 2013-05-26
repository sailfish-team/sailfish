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

#include <limits>
#include <climits>
#include <cassert>
#include <utility>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include <cvmlcpp/base/stl_cmath.h>

#include <cvmlcpp/base/use_omp.h>
#include <omptl/omptl_algorithm>
#include <omptl/omptl_numeric>

#include <cvmlcpp/base/Functors>
#include <cvmlcpp/base/CyclicBuffer>

#include <cvmlcpp/math/ContainerOps.h>

namespace cvmlcpp
{

namespace detail
{

template <typename T, typename U>
T round_cast_(const U &u, std::tr1::true_type)
{
	return static_cast<T>(u);
}

template <typename T, typename U>
T round_cast_(const U &u, std::tr1::false_type)
{
	return static_cast<T>(u + ((u>=0)?U(0.5):U(-0.5)) );
}

} // end namespace detail

template <typename T, typename U>
T round_cast(const U &u)
{
	return detail::round_cast_<T, U>(u, std::tr1::is_floating_point<T>());
}

namespace detail
{

template <typename T>
T log2_(T n, std::tr1::false_type)
{
	assert(n > 0);
	const std::size_t N = CHAR_BIT*sizeof(T);

	T result = 0;
	for (std::size_t i = 1; i < N; ++i)
	{
		const std::size_t M = N-i;
		if ( n >= (std::size_t(1) << M) )
		{
			n >>= M;
			result |= M;
		}
	}

	return result;
}

template <typename T>
T log2_(const T &n, std::tr1::true_type)
{
#ifdef _HAVE_TR1_CMATH
	using std::tr1::log2;
	return log2(n);
#else
	using std::log;
	return log(n) / log(2.0);
#endif
}

} // end namespace detail

template <typename T>
T log2(const T n)
{
	assert(n > T(0)); // Bogus user input ?
	return detail::log2_(n, std::tr1::is_floating_point<T>());
}

template <typename T>
T factorial(const T &x)
{
	// Check user input
	assert(x >= 0);

	static std::size_t ntop = 12;
	const std::size_t cache_size = 32;
	static T cache[cache_size] = {
	1, 1, 2, 6, 24, 120, 720, 5040,
	40320, 362880, 3628800, 39916800, 479001600};

#ifdef _HAVE_TR1_CMATH
	assert(x == std::tr1::round(x));
#endif

	if (x >= T(cache_size))
	{
		#ifdef _HAVE_TR1_CMATH
 		return std::tr1::round(std::tr1::tgamma(x+1.0));
		#else
		assert(ntop > 0);
		T v = cache[ntop-1];
		std::size_t part1 = round_cast<std::size_t>(std::min(T(cache_size), x));
		for (std::size_t i = ntop+1; i < part1; ++i)
		{
			v *= static_cast<T>(i);
			cache[i] = v;
		}
		ntop = part1;

		// Non-cached part
		for (T i = part1+1; i < x; ++i)
			v *= static_cast<T>(i);

		return v;
		#endif
	}

	while (x > ntop)
	{
		cache[ntop] = cache[ntop-1u] * ntop;
		++ntop;
	}
	assert(ntop >= x);

#ifdef _HAVE_TR1_CMATH
	return cache[std::tr1::lround(x)];
#else
	return cache[round_cast<std::size_t>(x)];
#endif
}

// Function gcd adapted from wikipedia: http://en.wikipedia.org/wiki/Binary_GCD_algorithm
namespace detail
{

template <bool is_integer = false>
struct GCD
{
	// euclidean based implementation
	template <typename T>
	static const T gcd(T u, T v)
	{
		assert(u >= 0);
		assert(v >= 0);

		using std::floor;
		u = floor(u+T(0.5));
		v = floor(v+T(0.5));

		using std::max;
		if (u < 1 || v < 1) // in case one is zero
			max(u, v);

		while (u > T(0.5)) // take care of roundoff problems
		{
			const T t = u;
			using std::floor;
			u = floor(v - floor(v / u)*u + T(0.5)); // modulus
			v = t;
		}
		assert(floor(v) == v);

		return v;
	}
};

template <>
struct GCD<true>
{
	// Function gcd adapted from wikipedia: http://en.wikipedia.org/wiki/Binary_GCD_algorithm
	template <typename T>
	static const T gcd(T u, T v)
	{
		// Use this function only for integers
		assert(std::tr1::is_integral<T>::value);

		/* GCD(0,x) := x */
		if (u == 0 || v == 0)
			return u | v;

		/* Let shift := lg K, where K is the greatest power of 2
		dividing both u and v. */
		T shift = 0;
		for (; ((u | v) & 1) == 0; ++shift)
		{
			u >>= 1;
			v >>= 1;
		}

		while ((u & 1) == 0)
			u >>= 1;

		/* From here on, u is always odd. */
		assert(u & 1);
		do {
			while ((v & 1) == 0)  /* Loop X */
				v >>= 1;

			/* Now u and v are both odd, so diff(u, v) is even.
			Let u = min(u, v), v = diff(u, v)/2. */
			if (u < v)
				v -= u;
			else
			{
				const T diff = u - v;
				u = v;
				v = diff;
			}
			v >>= 1;
		} while (v != 0);

		return u << shift;
	}
};

}

template <typename T>
T gcd(const T u, const T v)
{
	return detail::GCD<std::tr1::is_integral<T>::value>::gcd(u, v);
}

namespace detail
{

template <bool is_integer = false>
struct Binomial
{
	template <typename T>
	static const T binomial(const T n, const T k)
	{
		//return factorial(n) / (factorial(k) * factorial(n-k));
		if (n == k)
			return 1;
		if (n == k+1)
			return n;

		// Factors
		assert(k < n);
		std::vector<T> ns(n-k), ks(n-k-1);
		assert(ks.size() >= 1);
		for (T i = 0; i < n-k; ++i)
			ns[i] = k+1+i;
		for (T i = 2; i <= n-k; ++i)
			ks[i-2] = i;

		// Simplify
		for (std::size_t i = 0; i < ks.size(); ++i)
		{
			if (ks[i] > 1)
			{
				for (std::size_t j = 0; j < ns.size(); ++j)
					if (ns[j] > 1)
					{
						const T d = gcd(ns[j], ks[i]);
//						assert( (ks[i] % d) == 0 );
//						assert( (ns[j] % d) == 0 );
						ks[i] /= d;
						ns[j] /= d;
					}
			}
		}

		// After simplification, denominator is one
		assert(std::abs(std::accumulate(ks.begin(), ks.end(), T(1), std::multiplies<T>()) - T(1)) < 0.001);

		// Final division not needed, denominator is one
		return std::accumulate(ns.begin(), ns.end(), T(1), std::multiplies<T>());
	}
};

template <>
struct Binomial<true>
{
	template <typename T>
	static const T binomial(const T n, T k)
	{
		enum {N, K, RESULT};
		const T CACHE_SIZE = 29;
		static std::vector<std::vector<std::tr1::array<T, 3> > > cache;

		// Init cache if needed
		if (cache.empty())
		{
			cache.resize(CACHE_SIZE);
			for (T i = 0; i < CACHE_SIZE; ++i)
			{
				cache[i].resize(CACHE_SIZE);
				for (T j = 0; j < CACHE_SIZE; ++j)
					std::fill(cache[i][j].begin(), cache[i][j].end(), 0);
			}
		}

		// Check in cache
		const T cn = n % CACHE_SIZE;
		const T ck = k % CACHE_SIZE;
		if ( (cache[cn][ck][N] == n) && (cache[cn][ck][K] == k) )
			return cache[cn][ck][RESULT];


		/*
		 * General approach:
		 *      n!       prod(k+1:n)
		 * ---------- = ----------
		 * (n-k)! k!      (n-k)!
		 * Then factor out GCD's to try to avoid overflow
		 */
		assert (k <= n);

		// Optimize for complexity, binomial is symmetric for
		// k or n-k, but selecting the largest of the 2 leads to
		// less terms in the sequences.
		k = std::max(k, n-k);

		// Simple cases that lead to complex indices below
		if (n == k)
			return 1;
		if (n == k+1)
			return n;

		// Factors
		assert(k < n);
		std::vector<T> ns(n-k), ks(n-k-1);
		assert(ks.size() >= 1);
		for (T i = 0; i < n-k; ++i)
			ns[i] = k+1+i;
		for (T i = 2; i <= n-k; ++i)
			ks[i-2] = i;

		// Simplify
		for (std::size_t i = 0; i < ks.size(); ++i)
		{
			if (ks[i] > 1)
			{
				for (std::size_t j = 0; j < ns.size(); ++j)
					if (ns[j] > 1)
					{
						const T d = gcd(ns[j], ks[i]);
						assert( (ks[i] % d) == 0 );
						assert( (ns[j] % d) == 0 );
						ks[i] /= d;
						ns[j] /= d;
					}
			}
		}

		// After simplification, denominator is one
		assert(std::accumulate(ks.begin(), ks.end(), T(1), std::multiplies<T>()) == T(1));

		// Final division not needed, denominator is one
		const T result = std::accumulate(ns.begin(), ns.end(), T(1), std::multiplies<T>());
				// / std::accumulate(ks.begin(), ks.end(), T(1), std::multiplies<T>());

		// keep result in cache
		cache[cn][ck][RESULT] = result;

		return result;
	}
};

}

template <typename T>
T binomial(const T n, const T k)
{
	// verify user input
	assert(n >= k);
	return detail::Binomial<std::tr1::is_integral<T>::value>::binomial(n, k);
}

template <typename T>
T binopmf(const std::size_t n, const std::size_t k, const T p)
{
	BOOST_STATIC_ASSERT(!std::numeric_limits<T>::is_integer);
	assert(n >= k);
	assert(p >= 0);
	assert(p <= 1);
	using std::pow;

	const std::size_t max_bits_needed_for_binomial = (n+1)/2 * (log2(n)+1);

	const T binomial_n_k = (max_bits_needed_for_binomial < CHAR_BIT * sizeof(std::size_t) ) ?
		binomial(n, k) : // fast integer binomial
		binomial(T(n), T(k));  // FP binomial, no overflow

	return binomial_n_k * pow(p, T(k)) * pow( T(1)-p, T(n-k) );
}

template <typename T>
T binocdf(const std::size_t n, const std::size_t k, const T p)
{
	BOOST_STATIC_ASSERT(!std::numeric_limits<T>::is_integer);
	assert(n >= k);
	assert(p >= 0);
	assert(p <= 1);

	T bin = 1;
	T result = 0;
	T pk = 1;
	using std::pow;
	T pnk = pow(T(1)-p, T(n));
	for (std::size_t i = 0; i < k; ++i)

	{
		result += T(bin) * pk * pnk;
		//const T g = gcd(n-i, i+1);
		bin *= (n-i);  //((n-i) / g);
		bin /= (i+1);  //((i+1) / g);
//		assert(bin == binomial(n, i+1)); // can be triggered by numerical issues
		pk  *= p;
		pnk /= T(1)-p;
	}
	result += bin * pk * pnk;

	return result;
}

template <typename T, typename U>
std::size_t binocdfinv(const T p_arg, const std::size_t n, const U p)
{
	BOOST_STATIC_ASSERT(!std::numeric_limits<T>::is_integer);
	assert(n > 0);

	assert(p_arg >= 0.0);
	assert(p_arg <= 1.0);

	assert(p >= 0.0);
	assert(p <= 1.0);

	std::size_t result = 0;
	T cdf = binopmf(0, n, p);
	while ( (cdf < p_arg) && (result < n) )
	{
		++result;
		cdf += binopmf(result, n, p);
	}
	assert(result <= n);
	return result;
}

#ifdef __GXX_EXPERIMENTAL_CXX0X__
template <typename T>
T qfunc(const T x) { return T(0.5) * std::erfc(x / std::sqrt(T(2))); }
#endif

/**
 * Simple Statistics
 */

namespace detail
{

template <typename T>
typename T::value_type modulus_(const T &x, std::tr1::false_type)
{
	typedef typename T::value_type VT;
	const VT rv = omptl::transform_accumulate(x.begin(), x.end(), VT(0),
						cvmlcpp::Square<VT>());
	return round_cast<VT>(std::sqrt(rv));
}

template <typename T>
T modulus_arithmic_(const T &x, std::tr1::true_type)
{
	return x;
}

template <typename T>
T modulus_arithmic_(const T &x, std::tr1::false_type)
{
	using std::abs;
	return abs(x);
}

template <typename T>
T modulus_(const T &x, std::tr1::true_type)
{
	return modulus_arithmic_(x, std::tr1::is_unsigned<T>());
}

} // end namespace detail

template <typename T>
typename ValueType<T>::value_type modulus(const T &x)
{
	return round_cast<typename ValueType<T>::value_type>
		(detail::modulus_(x, std::tr1::is_arithmetic<T>()));
}

template <typename Iterator, typename T>
typename promote_trait1<T>::value_type average(Iterator begin, Iterator end, const T init)
{
	typedef typename promote_trait1<T>::value_type R;
	const typename std::iterator_traits<Iterator>::difference_type
		n = std::distance(begin, end);
	assert(n >= 0);

	if (n == 0)
		return R(init);

	return R(omptl::accumulate(begin, end, init)) / static_cast<R>(n);
}

template <typename Iterator, typename T>
T median(Iterator begin, Iterator end, const T init)
{
	if (std::distance(begin, end) < 1)
		return init;
	std::vector<T> data(std::distance(begin, end));
	std::copy(begin, end, data.begin());
	omptl::sort(data.begin(), data.end());
	return init + *(data.begin() + data.size() / 2u);
}

namespace detail
{

template <typename T>
class Variance_
{
	public:
		typedef T value_type;
		typedef T result_type;

		Variance_(const T mean = T(0)) : mean_(mean) {}

		T operator()(const T &elem) const
		{
			return std::pow(elem - mean_, T(2));
		}

	private:
		const T mean_;
};

} // end namespace detail

template <typename Iterator, typename T, typename U>
typename promote_trait1<T>::value_type variance(Iterator begin, Iterator end, const T mean, const U init)
{
	typedef typename promote_trait1<T>::value_type R;
	const typename std::iterator_traits<Iterator>::difference_type
		n = std::distance(begin, end);
	assert(n >= 0);

	if (n < 2)
		return init;

	if (mean == T(0))
		return R(init) +
			R(omptl::transform_accumulate(begin,end,T(0),Square<T>()))
			// ----------------------------------------------
					/ static_cast<R>(n-1);

	detail::Variance_<T> var((mean));
	return R(init) + R(omptl::transform_accumulate(begin, end, T(0), var))
			// ----------------------------------------------
					/ static_cast<R>(n-1);
}

template <typename Iterator, typename T>
typename promote_trait1<T>::value_type variance(Iterator begin, Iterator end, const T init)
{
	return variance(begin, end, average(begin, end, T(0)), init);
}

template <typename Iterator, typename T>
typename promote_trait1<T>::value_type deviation(Iterator begin, Iterator end, const T init)
{
	return init + std::sqrt(variance(begin, end, T(0), init));
}

template <typename Iterator, typename T, typename U>
typename promote_trait1<T>::value_type deviation(Iterator begin, Iterator end, const T mean, const U init)
{
	return init + std::sqrt(variance(begin, end, mean, U(0)));
}

template <typename Iterator, typename T>
T generalizedGaussianShape(Iterator begin, Iterator end,
			   const T avg, const T init)
{
	const typename std::iterator_traits<Iterator>::difference_type
		n = std::distance(begin, end);
	assert(n >= 0);

	if (n < 2)
		return init;

	const T lut [] = {
		0.0000000e+00, 1.3426932e-03, 1.1876961e-02, 3.5319079e-02,
		6.7911558e-02, 1.0500095e-01, 1.4333004e-01, 1.8099448e-01,
		2.1699493e-01, 2.5087260e-01, 2.8247508e-01, 3.1181684e-01,
		3.3899861e-01, 3.6416155e-01, 3.8746151e-01, 4.0905505e-01,
		4.2909201e-01, 4.4771200e-01, 4.6504301e-01, 4.8120124e-01,
		4.9629158e-01, 5.1040840e-01, 5.2363643e-01, 5.3605172e-01,
		5.4772256e-01, 5.5871029e-01, 5.6907008e-01, 5.7885162e-01,
		5.8809974e-01, 5.9685492e-01, 6.0515382e-01, 6.1302967e-01,
		6.2051264e-01, 6.2763020e-01, 6.3440738e-01, 6.4086703e-01,
		6.4703005e-01, 6.5291558e-01, 6.5854121e-01, 6.6392307e-01,
		6.6907603e-01, 6.7401380e-01, 6.7874902e-01, 6.8329340e-01,
		6.8765777e-01, 6.9185215e-01, 6.9588588e-01, 6.9976761e-01,
		7.0350541e-01, 7.0710678e-01, 7.1057873e-01, 7.1392780e-01,
		7.1716011e-01, 7.2028138e-01, 7.2329696e-01, 7.2621189e-01,
		7.2903089e-01, 7.3175837e-01, 7.3439853e-01, 7.3695527e-01,
		7.3943229e-01, 7.4183308e-01, 7.4416092e-01, 7.4641892e-01,
		7.4861001e-01, 7.5073697e-01, 7.5280242e-01, 7.5480885e-01,
		7.5675862e-01, 7.5865396e-01, 7.6049699e-01, 7.6228974e-01,
		7.6403410e-01, 7.6573191e-01, 7.6738490e-01, 7.6899471e-01,
		7.7056292e-01, 7.7209102e-01, 7.7358045e-01, 7.7503256e-01,
		7.7644865e-01, 7.7782996e-01, 7.7917769e-01, 7.8049297e-01,
		7.8177687e-01, 7.8303045e-01, 7.8425468e-01, 7.8545054e-01,
		7.8661892e-01, 7.8776070e-01, 7.8887673e-01, 7.8996780e-01,
		7.9103468e-01, 7.9207813e-01, 7.9309885e-01, 7.9409752e-01,
		7.9507481e-01, 7.9603134e-01, 7.9696773e-01, 7.9788456e-01 };

	const detail::Variance_<T> var;
	if (avg != 0.0)
	{
		// Must substract mean ...
		std::vector<T> zeroMeanData( (std::distance(begin, end)) );
		omptl::transform(begin, end, zeroMeanData.begin(),
				 std::bind2nd(std::plus<T>(), -avg) );

		const T sig = std::sqrt(
			omptl::transform_accumulate(zeroMeanData.begin(),
					    zeroMeanData.end(), T(0), var)
			// ----------------------------------------------
					/ static_cast<T>(n-1) );

		const T mean_abs =
			omptl::transform_accumulate(zeroMeanData.begin(),
						    zeroMeanData.end(), T(0),
				 std::ptr_fun( ( (T(*)(T))std::abs) ) )
			// ----------------------------------------------
					/ static_cast<T>(n);

		const T* pos = std::lower_bound(lut, lut+100, mean_abs/sig );

		return init + T(std::distance(lut, pos)) * 0.02;
	}

	// Avg == 0.0; faster calculation

	const T sig =std::sqrt(
			omptl::transform_accumulate(begin, end, T(0), var)
		// ----------------------------------------------
				/ static_cast<T>(n-1) );

	const T mean_abs =
		omptl::transform_accumulate(begin, end, T(0),
					    std::ptr_fun( (T(*)(T))std::abs ) )
		// ----------------------------------------------
				/ static_cast<T>(n);

	const T* pos = std::lower_bound(lut, lut+100, mean_abs/sig );

	return init + T(std::distance(lut, pos)) * 0.02;
}

template <typename Iterator, typename T>
T generalizedGaussianShape(Iterator begin, Iterator end, const T init)
{
	return generalizedGaussianShape(begin, end,
					average(begin, end, T(0)), init);
}

/*
 * Creative optimalization.
 * No idea if this is supposed to work
 */

namespace detail
{

template <class T>
struct OptimizationUpdater;

template <bool value_is_arithmetic, bool result_is_arithmetic>
struct OptimizationUpdater_
{
	template <class value_type, class result_type>
	static bool update(value_type &low,  value_type &dLow,  result_type &fLow,//  value_type &bLow,
			   value_type &high, value_type &dHigh, result_type &fHigh,// value_type &bHigh,
			   const value_type &x, const value_type &dx, const result_type &fx)
	{
		assert(dLow * dHigh <= 0);
		if (dx*dLow > 0) // same sign ?
		{
			assert (x >= low);
			assert(dx * dHigh <= 0); // need opposing signs
			low  = x;
			dLow = dx;
			fLow = fx;
//			bLow  = fLow - dLow  * low;
		}
		else if (dx*dHigh > 0)
		{
			assert (x <= high);
			assert(dx * dLow <= 0); // need opposing signs
			high  = x;
			dHigh = dx;
			fHigh = fx;
//			bHigh = fHigh - dHigh * high;
		}
		else
		{
/*			std::cout << std::scientific;
			if (!(dx == 0))
			{
				std::cout << low << " " << x << " " << high << std::endl;
				std::cout << dLow << " " << dx << " " << dHigh << std::endl;
			}
*/
			assert(dx == 0);

/*			// just in case
			high  = low  = x;
			dHigh = dLow = dx;
			fHigh = fLow = fx;
*/
			return false; // derivative zero --> optimum found
		}
		assert(dLow * dHigh <= 0); // need opposing signs

		return true;
	}

};

template <>
struct OptimizationUpdater_<false, false>
{
	template <class value_type, class result_type>
	static bool update(value_type &low,  value_type &dLow,  result_type &fLow,//  value_type &bLow,
			   value_type &high, value_type &dHigh, result_type &fHigh,// value_type &bHigh,
			   const value_type &x, const value_type &dx, const result_type &fx)
	{
		typedef typename value_type::value_type VT;
		std::size_t n = 0;
		for (std::size_t i = 0; i < low.size(); ++i)
		{
			if (OptimizationUpdater< VT >::
			    update(low[i], dLow[i], fLow[i],// bLow[i],
				   high[i], dHigh[i], fHigh[i],// bHigh[i],
				   x[i], dx[i], fx[i]))
				++n;
		}

		// Update considered successful if at least one parameter was updated
		return n > 0;
	}
};

template <>
struct OptimizationUpdater_<true, false>
{
	template <class value_type, class result_type>
	static bool update(value_type &low,  value_type &dLow,  result_type &fLow,//  value_type &bLow,
			   value_type &high, value_type &dHigh, result_type &fHigh,// value_type &bHigh,
			   const value_type &x, const value_type &dx, const result_type &fx)
	{
		typedef typename value_type::value_type VT;
		std::size_t n = 0;
		for (std::size_t i = 0; i < low.size(); ++i)
		{
			if (OptimizationUpdater< VT >::
			    update(low, dLow, fLow[i],// bLow,
				   high, dHigh, fHigh[i],// bHigh,
				   x, dx, fx[i]))
				++n;
		}

		// Update considered successful if at least one parameter was updated
		return n > 0;
	}
};

template <>
struct OptimizationUpdater_<false, true>
{
	template <class value_type, class result_type>
	static bool update(value_type &low,  value_type &dLow,  result_type &fLow,//  value_type &bLow,
			   value_type &high, value_type &dHigh, result_type &fHigh,// value_type &bHigh,
			   const value_type &x, const value_type &dx, const result_type &fx)
	{
		typedef typename value_type::value_type VT;
		std::size_t n = 0;
		for (std::size_t i = 0; i < low.size(); ++i)
		{
			if (OptimizationUpdater< VT >::
			    update(low[i], dLow[i], fLow,// bLow[i],
				   high[i], dHigh[i], fHigh,// bHigh[i],
				   x[i], dx[i], fx))
				++n;
		}

		// Update considered successful if at least one parameter was updated
		return n > 0;
	}
};

template <class T>
struct OptimizationUpdater
{
	template <class value_type, class result_type>
	static bool update(value_type &low,  value_type &dLow,  result_type &fLow,//  value_type &bLow,
			   value_type &high, value_type &dHigh, result_type &fHigh,// value_type &bHigh,
			   const value_type &x, const value_type &dx, const result_type &fx)
	{
		return OptimizationUpdater_<	std::tr1::is_arithmetic< value_type>::value,
						std::tr1::is_arithmetic<result_type>::value >::
			    update(low, dLow, fLow,/* bLow,*/ high, dHigh, fHigh,/* bHigh,*/ x, dx, fx);
	}
};

} // end namespace detail

template <class Function, class Derivative>
typename Function::value_type optimize(const Function &f, const Derivative &d,
		typename Function::value_type low,
		typename Function::value_type high,
		const std::size_t N)
{
	typedef typename Function  :: value_type  value_type;
	typedef typename Derivative::result_type result_type;

	// Calculate derivatives in end points
	value_type dLow  = d(low);  if (dLow  == 0) return low;
	value_type dHigh = d(high); if (dHigh == 0) return high;

	value_type x = (low + high) / value_type(2);
	assert(dLow * dHigh < 0); // need opposing signs
	if (!(dLow * dHigh < 0)) // Invalid input, actually
		return x;

	result_type fLow  = f(low);
	result_type fHigh = f(high);

//	value_type bLow  = fLow  - dLow  * low;
//	value_type bHigh = fHigh - dHigh * high;
/*	std::cout << "INIT" << std::endl;
	std::cout << "low: " << low << std::endl;
	std::cout << "dLow: " << dLow << std::endl;
	std::cout << "fLow: " << fLow << std::endl;
//	std::cout << "bLow: " << bLow << std::endl;
	std::cout << "high: " << high << std::endl;
	std::cout << "dHigh: " << dHigh << std::endl;
	std::cout << "fHigh: " << fHigh << std::endl;
//	std::cout << "bHigh: " << bHigh << std::endl;
 	std::cout << std::endl;
*/
	for (std::size_t i = 0; i < N; ++i)
	{
		assert(dLow * dHigh <= 0); // need opposing signs

		// find two lines y = d*x + b

		// find intersection point of derivatives as
		// estimation of center, find derivative at center
		//x = (bHigh - bLow) / (dLow - dHigh);
		x = (low + high) / value_type(2);
		assert(low <= x);
		assert(x <= high);
		const result_type fx = f(x);
		const  value_type dx = d(x);
/*		std::cout << "x: " << x << std::endl;
		std::cout << "fx: " << fx << std::endl;
		std::cout << "dx: " << dx << std::endl;
*/
		if (!detail::OptimizationUpdater< value_type >::
		    update(low, dLow, fLow,/* bLow,*/ high, dHigh, fHigh,/* bHigh,*/ x, dx, fx))
		{
			// No parameters updated, derivative zero --> optimum found
			break;
		}
/*		std::cout << "low: " << low << std::endl;
		std::cout << "dLow: " << dLow << std::endl;
		std::cout << "fLow: " << fLow << std::endl;
//		std::cout << "bLow: " << bLow << std::endl;
		std::cout << "high: " << high << std::endl;
		std::cout << "dHigh: " << dHigh << std::endl;
		std::cout << "fHigh: " << fHigh << std::endl;
//		std::cout << "bHigh: " << bHigh << std::endl;
		std::cout << std::endl;
*/
	}
	return x;
}

template <class Function>
typename Function::value_type optimize(const Function &f,
			typename Function::value_type low,
			typename Function::value_type high,
			const std::size_t N)
{
	return optimize(f, Derivative<Function>(f), low, high, N);
}


/*
 * Newton-Raphson
 */

template <class Function, class Derivative>
bool doNewtonRaphson(const Function &f, const Derivative &d,
			typename Function::value_type &x, const std::size_t N)
{
	typedef typename Function::value_type value_type;
	typedef typename ValueType<value_type>::value_type T;
	CyclicBuffer<T> values((16u));

	value_type dx = value_type(T(1));
	T m = T(0);
	value_type fx = f(x);

	value_type oldx [2] = {x, x};
	value_type avgx = x;

	value_type best = x;
	value_type lowest = fx;

	// Start loop, terminate on convergence or hopelessness
	for (std::size_t i = 0u; i < N; ++i)
	{
// std::cout << i << " X: " << x << " f(x): " << fx << " df(x): " << d(x)
// 	<< " dx: " << dx << " avg: " << values.avg() << std::endl;

		if (modulus(fx) == T(0))
			return true;

		// The method 2nd-order, i.e. error should be quadratic. Hence,
		// the updates to x should reduce at 2nd order as well, and thus
		// allways be smaller than previous updates, at least on
		// average.
		// Convergence is reached when the updates no longer become
		// smaller and smaller. To have a meaningful average value,
		// run at least "values.capacity()" times.
		if ( (i > values.capacity()) && (values.avg() <= m) )
			break;

		values.add(m);			// Add metric _after_ comparison

		// Compute standard Newton-Raphson, but avoid divide-by-zero.
		value_type next = x - dx; 	// if df==0, repeat previous
		const value_type df = d(x);	// Compute derivative
		if (modulus(df) > T(0))		// Not reliable for Vectors...
			next = x - fx / df;

		// Select best option of standard Newton-Raphson and
		// bisection of previous values.
		value_type fnext = f(next);
		const value_type favgx = f(avgx);
		if (modulus(favgx) < modulus(fnext))
		{
			next = avgx;
			fnext= favgx;
		}

		dx   = next - x;		// Displacement
		avgx = (oldx[0] + oldx[1]) / T(2); // bisection of previous
		oldx[i%2u] = x;
		x    = next;
		fx   = fnext;
		m    = modulus(dx);		// Quality metric = abs(update)

		// Select best result.
		if (modulus(fx) < modulus(lowest))
		{
			best   = x;
			lowest = fx;
		}
	}
	x = best;

	return (values.avg() <= m); // Converged or ran out of iterations ?
}

template <class Function>
bool doNewtonRaphson(const Function &f, typename Function::value_type &x,
			const std::size_t N)
{
	return doNewtonRaphson(f, Derivative<Function>(f), x, N);
}

template <class Function, class Derivative>
bool doNewtonRaphson(const Function &f, const Derivative &d,
			typename Function::value_type &x,
			const typename Function::value_type low,
			const typename Function::value_type high,
			const std::size_t N)
{
	typedef typename Function::value_type value_type;
	typedef typename ValueType<value_type>::value_type T;

	// Make sure x1 < x2
	value_type xl = std::min(low, high);
	value_type xh = std::max(low, high);

	if ( (xl > x) || (x > xh) )
		x = (xl + xh) / T(2);

	CyclicBuffer<T> values((16u));

	T m = T(0);
	value_type dx	= T(0);
	value_type fx	= f(x);
	value_type fl	= f(xl);
	value_type fh	= f(xh);

	value_type oldx [2] = {x, x};
	value_type avgx = x;

	value_type best = x;
	value_type lowest = fx;

	// Start loop, terminate on convergence or hopelessness
	for (std::size_t i = 0u; i < N; ++i)
	{
// std::cout << i << " X: " << x << " f(x): " << fx << " df(x): " << d(x)
// 	<< " dx: " << dx << " avg: " << values.avg()
// 	<< " [" << xl << ", " << xh << "]"<< std::endl;

		if (modulus(fx) == T(0)) // Jackpot ?
			return true;

		// The method 2nd-order, i.e. error should be quadratic. Hence,
		// the updates to x should reduce at 2nd order as well, and thus
		// allways be smaller than previous updates, at least on
		// average.
		// Convergence is reached when the updates no longer become
		// smaller and smaller. To have a meaningful average value,
		// run at least "values.capacity()" times.
		if ( (i > values.capacity()) && (values.avg() <= m) )
			break;

		values.add(m);		// Add metric _after_ comparison

		T next = x + dx;
		const value_type df = d(x);

		if (modulus(df) > T(0)) // Not reliable for Vectors...
			next = x - fx / df;
		next = clamp(next, xl, xh);

		// Select an alternative option as best of
		// bi-section of range and bisection of previous values.
		value_type fnext = f(next);
		value_type alternative  = (xh + xl) / T(2); // bisect of range
		value_type falternative = f(alternative);

// std::cout << "\t next: " << next << " middle: " << alternative << " avgx " <<
// 	avgx << " fnext "<< fnext << " falternative " << falternative <<
// 	" favg " << favgx << std::endl;

		const value_type favgx = f(avgx); // bisect of previous values
		if (modulus(favgx) < modulus(falternative))
		{
			alternative = avgx;
			falternative= favgx;
		}

		// Select best option of standard Newton-Raphson and alternative
		if ( !((xl <= next) && (next <= xh)) ||
		      (modulus(falternative) < modulus(fnext)) )
		{
			next = alternative;
			fnext= falternative;
		}

		dx	= next - x;		// Displacement
		m	= modulus(dx);		// Quality metric = abs(update)
		avgx = (oldx[0] + oldx[1]) / T(2); // bisection of previous
		oldx[i%2u] = x;
		x	= next;			// Update position
		fx	= fnext;		// Update function value

		// Try to reduce the range such that the function has one
		// end-point below zero and one above zero.
		if (fl * fx < T(0))
		{
			xh = x;
			fh = fx;
		}
		if (fh * fx < T(0))
		{
			xl = x;
			fl = fx;
		}

		// Select the best option
		if (modulus(fx) < modulus(lowest))
		{
			best   = x;
			lowest = fx;
		}
	}
	x = best;

	return (values.avg() <= m);  // Converged or ran out of iterations ?
}

template <class Function>
bool doNewtonRaphson(const Function &f, typename Function::value_type &x,
			const typename Function::value_type low,
			const typename Function::value_type high,
			const std::size_t N)
{
	return doNewtonRaphson(f, Derivative<Function>(f),
				x, low, high, N);
}


template <class Function>
void doRungeKutta(const Function &f,
		const typename Function::value_type t0,
		const typename Function::value_type tN,
		const std::size_t N,
		const typename Function::value_type y0,
		std::vector<typename Function::value_type> &y)
{
	assert(tN > t0); // Bogus user input ?

	typedef typename Function::value_type T;
	assert(std::tr1::is_floating_point<T>::value); // Float types only.

	y.clear();
	y.reserve(N+1u); // Ensure 1 memory alloc only

	// "physical" spacing between two nodes
	const T h = (tN - t0) / static_cast<T>(N);

	assert(T(0.5) * h > T(0.)); // Numerically still stable ?

	y.push_back(y0);          // initial condition
        for (std::size_t i = 0u; i < N; ++i)
        {
		assert(i == y.size() - 1u);

		const T t_n	= t0 + static_cast<T>(i+1u) * h;
		const T halfH	= T(0.5) * h;

		const T k1 = f(t_n,           y[i]);
		const T k2 = f(t_n + halfH,   y[i] + halfH * k1);
		const T k3 = f(t_n + halfH,   y[i] + halfH * k2);
		const T k4 = f(t_n + h,       y[i] + h * k3);

		y.push_back(y[i] + h / T(6) * (k1 + T(2)*k2 + T(2)*k3 + k4));
        }
	assert(y.size() == N+1u);
}


template <template <typename Tm, std::size_t D, typename Aux> class Array_t,
	  typename T, typename A>
void identity_matrix(Array_t<T, 2u, A> &m, const std::size_t N)
{
	typedef array_traits<Array_t, T, 2u, A>	AT;

	const std::size_t sz [] = {N, N};
	AT::resize(m, sz);
	omptl::fill(m.begin(), m.end(), T(0));
#ifdef _OPENMP
	#pragma omp parallel for
#endif
	for (int i = 0; i < int(N); ++i)
		m[i][i] = T(1);
}


/*
 * Adapated from:
 * http://www.crystalclearsoftware.com/cgi-bin/boost_wiki/wiki.pl?LU_Matrix_Inversion
 *
 * Matrix inversion routine.
 * Uses lu_factorize and lu_substitute in uBLAS to invert a matrix
 */

template<class T, class F, class A>
bool invert(boost::numeric::ublas::matrix<T, F, A>& m)
{
 	namespace ublas = boost::numeric::ublas;

 	// create a working copy of the input
 	ublas::matrix<T> a(m);

 	// create a permutation matrix for the LU-factorization
 	ublas::permutation_matrix<std::size_t> pm(a.size1());

 	// perform LU-factorization
 	if (ublas::lu_factorize(a, pm)) // returns zero on success
		return false;

	// create identity_matrix matrix of "inverse"
 	m.assign(ublas::identity_matrix<T>(a.size1()));

 	// backsubstitute to get the inverse
 	ublas::lu_substitute(a, pm, m);

 	return true;
}

template <template <typename Tm, std::size_t D, typename Aux> class Array_t,
	  typename T, typename A>
bool invert(Array_t<T, 2u, A> &m)
{
	typedef Array_t<T, 2u, A> Matrix_t;
	typedef array_traits<Array_t, T, 2u, A>	AT;

	// Must be NxN matrix
	if (AT::shape(m)[Y] != AT::shape(m)[X])
		return false;

	const std::size_t dim = AT::shape(m)[X];
	if (dim == 0)
		return true;

	// Load identity_matrix
	Matrix_t inv;
	cvmlcpp::identity_matrix(inv, dim);
	assert(AT::shape(inv)[X] == dim);
	assert(AT::shape(inv)[Y] == dim);

	for (std::size_t x = 0; x < dim; ++x)
	{
		// Scale
		if (m[x][x] == T(0))
			return false;

		const T scale = T(1) / m[x][x];
		m[x][x] = T(1);

#ifdef _OPENMP
		#pragma omp parallel for
#endif
		for (int y = x+1; y < int(dim); ++y)
			m[x][y]   *= scale;
#ifdef _OPENMP
		#pragma omp parallel for
#endif
		for (int y = 0; y < int(dim); ++y)
			inv[x][y] *= scale;

		// Sweep
#ifdef _OPENMP
		#pragma omp parallel for
#endif
		for (int xx = x+1; xx < int(dim); ++xx)
		{
			const T factor = m[xx][x];
			if (factor == T(0))
				continue;

			m[xx][x] = T(0);
			for (std::size_t yy = x+1; yy < dim; ++yy)
			{
				assert(x >= 0);
				assert(xx>= 0);
				assert(yy >= 0);
				assert(x < dim);
				assert(std::size_t(xx)< dim);
				assert(yy < dim);
				  m[xx][yy] -= factor *   m[x][yy];
			}
			for (std::size_t yy = 0;   yy < dim; ++yy)
			{
				assert(x >= 0);
				assert(xx>= 0);
				assert(yy >= 0);
				assert(x < dim);
				assert(std::size_t(xx)< dim);
				assert(yy < dim);
				inv[xx][yy] -= factor * inv[x][yy];
			}
		}
	}

	assert(dim > 0);
	for (long int x = dim - 1; x >= 0; --x)
	{
		// Sweep
#ifdef _OPENMP
		#pragma omp parallel for
#endif
		for (int xx = x-1; xx >= 0; --xx)
		{
			const T factor = m[xx][x];
			if (factor == T(0))
				continue;

			m[xx][x] = T(0);
			for (std::size_t y = x+1; y < dim; ++y)
			{
				assert(x >= 0);
				assert(xx>= 0);
				assert(y >= 0);
				assert(x < D);
				assert(xx< D);
				assert(y < D);
				m[xx][y] -= factor *   m[x][y];
			}
			for (std::size_t y = 0;   y < dim; ++y)
			{
				assert(x >= 0);
				assert(xx>= 0);
				assert(y >= 0);
				assert(x < D);
				assert(xx< D);
				assert(y < D);
				inv[xx][y] -= factor * inv[x][y];
			}
		}
	}

	m.swap(inv);

	return true;
}

template <template <typename Tm, std::size_t D, typename Aux> class Array_t,
	  typename T, typename A>
void transpose(Array_t<T, 2u, A> &m)
{
	typedef array_traits<Array_t, T, 2u, A>	AT;
	const std::size_t ext [] = {AT::shape(m)[Y], AT::shape(m)[X]};

	if ( (AT::size(m) == 0) || (ext[X] == 0) || (ext[Y] == 0) )
		return;

	std::vector<bool> todo(AT::size(m), true);
	todo[0] = false;

	typename std::vector<bool>::const_iterator next;
	T temp = *AT::begin(m);
	while ( (next=std::find(todo.begin(), todo.end(), true)) != todo.end())
	{
		using std::swap;
		std::size_t src  = next - todo.begin();

		std::size_t x1 = src / AT::shape(m)[Y];
		std::size_t y1 = src % AT::shape(m)[Y];
		swap(x1, y1);
		std::size_t dest = x1 * ext[Y] + y1;
		temp = *(AT::begin(m) + src);
		if (src == dest)
			todo[dest] = false;
		else
		{
			while (  todo[dest] )
			{
				typename AT::iterator destIt = AT::begin(m);
				std::advance(destIt, dest);

				swap(temp, *destIt);
				todo[dest] = false;
				src = dest;
				std::size_t x = src / AT::shape(m)[Y];
				std::size_t y = src % AT::shape(m)[Y];
				swap(x, y);
				dest = x * ext[Y] + y;
			}
		}
	}

	AT::resize(m, ext);

// 	for (std::size_t i = 0u; i < ext[1]; ++i)
// 	for (std::size_t j = 0u; j < ext[0]; ++j)
// 		r[i][j] = static_cast<T>(m[j][i]);
// 	r.swap(m);
}

namespace detail
{

template <template <typename Tm, std::size_t Dm, typename A> class Array_t,
	  typename Ta, typename Aux, class XVector_t, class YVector_t>
bool leastSquaresFit_ublas(const Array_t<Ta, 2, Aux> &A, const YVector_t &y,
		     XVector_t &x)
{
	// Copy to ublas matrix: A
	boost::numeric::ublas::matrix<Ta> A2 (A.extents()[X], A.extents()[Y]);
	for (std::size_t i = 0; i < A2.size1 (); ++i)
	for (std::size_t j = 0; j < A2.size2 (); ++j)
		A2(i, j) = A[i][j];

	// transposeA A^T
	const boost::numeric::ublas::matrix<Ta> At =
		boost::numeric::ublas::trans(A2);

	// Multiply, before invert: A^T A
	boost::numeric::ublas::matrix<Ta> AtAinv = //At * A2;
		boost::numeric::ublas::prod(At, A2);

/*
	std::cout << "AtA" << std::endl;
	for (std::size_t i = 0u; i < AtAinv.size1(); ++i)
	{
		for (std::size_t j = 0u; j < AtAinv.size2(); ++j)
	    		std::cout << AtAinv(i,j) << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
*/

	// Invert: (A^T A)^{-1}
	if (!invert(AtAinv))
		return false;

/*
	std::cout << "AtAinv" << std::endl;
	for (std::size_t i = 0u; i < AtAinv.size1(); ++i)
	{
		for (std::size_t j = 0u; j < AtAinv.size2(); ++j)
	    		std::cout << AtAinv(i,j) << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
*/

	// Copy observation vector
	boost::numeric::ublas::vector<Ta> y2 (y.size());
	for (std::size_t i = 0; i < y2.size (); ++ i)
		y2(i) = y[i];

	// x = (A^T A)^{-1} * At * y2;
	const boost::numeric::ublas::matrix<Ta> AtAinvAt  =
		boost::numeric::ublas::prod(AtAinv, At);
	const boost::numeric::ublas::vector<Ta> x2 =
		boost::numeric::ublas::prod(AtAinvAt, y2);

	assert(x.size() == x2.size());
	for (std::size_t i = 0; i < x.size (); ++ i)
		x[i] = x2(i);
}

template <template <typename Tm, std::size_t Dm, typename A> class Array_t,
	  typename Ta, typename Aux, class XVector_t, class YVector_t>
bool leastSquaresFit_cvmlcpp(const Array_t<Ta, 2, Aux> &A, const YVector_t &y,
		     XVector_t &x)
{
	typedef array_traits<Array_t, Ta, 2u, Aux> AT;
/*
	std::cout << "A" << std::endl;
	for (std::size_t i = 0u; i < A.extents()[X]; ++i)
	{
		for (std::size_t j = 0u; j < A.extents()[Y]; ++j)
	    		std::cout << A[i][j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
*/
	Array_t<Ta, 2, Aux> At = AT::copy_of(A);
	transpose(At);

	Array_t<Ta, 2, Aux> AtAinv; // = At * A;
	mat_mat_mult(At, A, AtAinv);
/*
	std::cout << "AtA" << std::endl;
	for (std::size_t i = 0u; i < AtAinv.extents()[X]; ++i)
	{
		for (std::size_t j = 0u; j < AtAinv.extents()[Y]; ++j)
	    		std::cout << AtAinv[i][j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
*/
	if (!invert(AtAinv))
		return false;
/*
	std::cout << "AtAinv" << std::endl;
	for (std::size_t i = 0u; i < AtAinv.extents()[X]; ++i)
	{
		for (std::size_t j = 0u; j < AtAinv.extents()[Y]; ++j)
	    		std::cout << AtAinv[i][j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
*/

	Array_t<Ta, 2, Aux> AtAinvAt;
	mat_mat_mult(AtAinv, At, AtAinvAt);
/*	std::cout << "B" << std::endl;
	for (std::size_t i = 0u; i < B.extents()[X]; ++i)
	{
		for (std::size_t j = 0u; j < B.extents()[Y]; ++j)
	    		std::cout << B[i][j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "Y" << std::endl;
	for (std::size_t i = 0u; i < y.size(); ++i)
		std::cout << y[i] << " ";
	std::cout << std::endl;
*/
	mat_vec_mult(AtAinvAt, y, x);

	return true;
}

} // end namespace detail

template <template <typename Tm, std::size_t Dm, typename A> class Array_t,
	  typename Ta, typename Aux, class XVector_t, class YVector_t>
bool leastSquaresFit(const Array_t<Ta, 2, Aux> &A, const YVector_t &y,
		     XVector_t &x)
{
	//return detail::leastSquaresFit_ublas(A, y, x);
	return detail::leastSquaresFit_cvmlcpp(A, y, x);
}

/*
 * The bits about the hungarian matching algorithm WAS part of
 * "the Stanford GraphBase (c) Stanford University 1993"
 * In accordance to the Stanford GraphBase license, this notice is here
 * to inform you that it is NO LONGER part of that library, and this file
 * has a different name than the one the original code was in, because the
 * code has been modified.
 */

namespace detail
{

// Subtract column minima in order to start with lots of zeroes
template <template <typename Tm, std::size_t D, typename Aux> class Matrix_type,
	typename T, typename A>
void to_zero(Matrix_type<T, 2, A> &aa)
{
	typedef array_traits<Matrix_type, T, 2, A> AT;
	const std::size_t m = AT::shape(aa)[0];
	const std::size_t n = AT::shape(aa)[1];

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int l = 0; l < int(n); ++l)
	{
		T s = aa[0][l];
		for (std::size_t k = 1; k < m; ++k)
			s = std::min(aa[k][l], s);
		if (s != 0)
			for (std::size_t k = 0; k < m; ++k)
				aa[k][l] -= s;
	}
}


template <template <typename Tm, std::size_t D, typename Aux> class Matrix_type,
	typename T, typename A>
void hungarian( const Matrix_type<T, 2, A> &aa,
		std::vector<std::ptrdiff_t> &col_mate,
		std::vector<std::ptrdiff_t> &row_mate)
{
//	printf("1\n");
	typedef array_traits<Matrix_type, T, 2, A> AT;

	/* number of rows and columns desired */
	const std::size_t m = AT::shape(aa)[0];
	const std::size_t n = AT::shape(aa)[1];

	col_mate.resize(m);
	row_mate.resize(n);
	std::vector<std::ptrdiff_t> parent_row(n);
	std::vector<std::ptrdiff_t> unchosen_row(m);
	std::vector<T> row_dec(m);
	std::vector<T> col_inc(n);
	std::vector<T> slack(n);
	std::vector<std::ptrdiff_t> slack_row(n);

	std::fill(row_mate  .begin(), row_mate  .end(), -1);
	std::fill(col_inc   .begin(), col_inc   .end(),  0);
	std::fill(parent_row.begin(), parent_row.end(), -1);
	std::fill(slack     .begin(), slack     .end(), std::numeric_limits<T>::max());
//	printf("2\n");

	/*
	 * The algorithm operates in stages, where each stage terminates
	 * when we are able to increase the number of matched elements.
	 *
	 * The first stage is different from the others; it simply goes through
	 * the matrix and looks for zeroes, matching as many rows and columns
	 * as it can. This stage also initializes table entries that will be
	 * useful in later stages.
	 */

	// total number of nodes in the forest
	std::size_t t = 0; /* the forest starts out empty */

	for (std::size_t k = 0; k < m; ++k)
	{
		// the minimum entry of row $k$
		const T s = *std::min_element(aa[k].begin(), aa[k].end());

		row_dec[k] = s;
		for (std::size_t l = 0; l < n; ++l)
			if ( (s==aa[k][l]) && (row_mate[l]<0))
			{
				col_mate[k]=l;
				row_mate[l]=k;
				goto row_done;
			}
		col_mate[k]=-1; // k unmatched
		unchosen_row[t++]=k;
//		printf("  node %ld: unmatched row %ld\n",t,k);
		row_done:;
		assert(t <= unchosen_row.size());
	}
//	printf("3\n");

	if (t==0)
		return;
	std::size_t unmatched = t;
	while(unmatched > 0)
	{
		std::ptrdiff_t k; /* the current row of interest */
		std::ptrdiff_t l; /* the current column of interest */
		/* the current matrix element of interest */

		T s;
//		printf("4\n");

//		std::size_t q = 0;
		while (1)
		{
			for (std::size_t q = 0; q < t; ++q)
			{
//				printf("5\n");
				// Explore node |q| of the forest;
				// if the matching can be increased, |goto breakthru|
				assert(q < m);
				k = unchosen_row[q];
//				const T s = row_dec[k];
				s = row_dec[k];
				for (l = 0; l < std::ptrdiff_t(n); ++l)
				{
//					printf("6\n");
					if (slack[l])
					{
						const T del = aa[k][l] - s + col_inc[l];
						if (del < slack[l])
						{
							if (del==0)
							{ /* we found a new zero */
								if (row_mate[l]<0)
									goto breakthru;
								slack[l] = 0; /* this column will now be chosen */
								parent_row[l] = k;
//								printf("  node %ld: row %ld==col %ld--row %ld\n", t,row_mate[l],l,k);
								assert(t < m);
								unchosen_row[t++] = row_mate[l];
							}
							else
							{
								slack[l] = del;
								slack_row[l] = k;
							}
						}
					}
				}
			}

//			fprintf(stderr, "7\n");

			// Introduce a new zero into the matrix by modifying |row_dec| and
			// |col_inc|; if the matching can be increased, |goto breakthru|
			/*T*/ s = std::numeric_limits<T>::max();
			for (std::size_t i = 0; i < n; ++i)
				if (slack[i] != 0)
					s = std::min(slack[i], s);
			assert(t <= unchosen_row.size());
			for (std::size_t q = 0; q < t; ++q) {
//				fprintf(stderr, "unchosen_row[%ld] = %ld\n", q, unchosen_row[q]);
				assert(unchosen_row[q] >= 0);
				assert(unchosen_row[q] < std::ptrdiff_t(row_dec.size()));
				row_dec[unchosen_row[q]] += s;
			}
			for (/*std::size_t*/ l = 0; l < std::ptrdiff_t(n); ++l)
			{
//				fprintf(stderr, "8\n");
				if (slack[l])
				{
					// column $l$ is not chosen
					slack[l] -= s;
					if (slack[l]==0)
					{
						// Look at a new zero, and |goto breakthru| with
						// |col_inc| up to date if there's a breakthrough
						k = slack_row[l];
//						fprintf(stderr, " Decreasing uncovered elements by %ld produces zero at [%ld,%ld]\n", s,k,l);
						if (row_mate[l]<0)
						{
							for (std::size_t j=l+1; j<n; ++j)
								if (slack[j] == 0)
									col_inc[j] += s;
							goto breakthru;
						}
						else
						{
							// not a breakthrough, but the forest continues to grow
							parent_row[l]=k;
//							fprintf(stderr, "  node %ld: row %ld==col %ld--row %ld\n", t,row_mate[l],l,k);
							unchosen_row[t++]=row_mate[l];
						}
					}
				}
				else
					col_inc[l]+=s;
			}
		}

		breakthru:
//		fprintf(stderr, "9\n");
		// Update the matching by pairing row $k$ with column $l$
		while (1)
		{
//			fprintf(stderr, "10\n");
			const std::ptrdiff_t j = col_mate[k];
			col_mate[k]=l;
			row_mate[l]=k;
//			fprintf(stderr, " rematching col %ld==row %ld\n",l,k);
			if (j < 0)
				break;
			k = parent_row[j];
			l = j;
		}

//		fprintf(stderr, "11\n");
		assert(unmatched > 0);
		if(--unmatched == 0)
			break;
		assert(unmatched > 0);

//		fprintf(stderr, "12\n");

		/*
		 *  get_ready_for_another_stage
		 */
		std::fill(parent_row.begin(), parent_row.end(), -1);
		std::fill(slack     .begin(), slack     .end(), std::numeric_limits<T>::max());
		t = 0;
		for (std::size_t i = 0; i < m; ++i)
			if (col_mate[i] < 0)
			{
				unchosen_row[t++] = i;
//				fprintf(stderr, "  node %ld: unmatched row %ld\n",t,k);
			}
	}
//	fprintf(stderr, "done!\n");
}

} // end namespace detail


/*
 * Interfacing.
 * A few nitty-gritty details still need to be handled: Our algorithm
 * is not symmetric between rows and columns, and it works only for $m\le n$;
 * so we will transpose the matrix when $m>n$. Furthermore, our
 * algorithm minimizes, but we actually want it to maximize (except
 * when |compl| is nonzero).
 *
 * Hence, we want to make the following transformations to the data
 * before processing it with the algorithm developed above.
 */

template <template <typename Tm, std::size_t D, typename Aux> class Matrix_type,
	typename T, typename A>
void find_matching(Matrix_type<T, 2, A> &costs,
		   std::vector<std::pair<std::size_t, std::size_t> > &matches,
		   const bool minimize_costs, const bool heuristic)
{
	typedef array_traits<Matrix_type, T, 2, A> AT;

	matches.clear();
	std::vector<std::ptrdiff_t> a_mate, b_mate;

	const std::size_t m = AT::shape(costs)[0];
	const std::size_t n = AT::shape(costs)[1];

	const bool transposed = m > n;
	if (transposed)
		transpose(costs);

	if (!minimize_costs)
	{
		const T max_cost = *omptl::max_element(costs.begin(), costs.end());
		omptl::transform(costs.begin(), costs.end(), costs.begin(),
					std::bind1st(std::minus<T>(), max_cost));
	}

	if ( heuristic && (m == n) )
		detail::to_zero(costs);
//fprintf(stderr, "GO!\n");
	detail::hungarian(costs, a_mate, b_mate);

	if (transposed)
		a_mate.swap(b_mate);

	assert(a_mate.size() <= b_mate.size());
	for (std::size_t a = 0; a < a_mate.size(); ++a)
		if (a_mate[a] >= 0)
		{
			// Check reprociatability
			assert(b_mate[a_mate[a]] == std::ptrdiff_t(a));
			matches.push_back( std::make_pair(a, a_mate[a]) );
		}
}

} // end namespace cvmlcpp
