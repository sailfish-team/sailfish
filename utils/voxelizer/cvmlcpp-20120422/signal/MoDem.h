/***************************************************************************
 *   Copyright (C) 2007,2008 by BEEKHOF, Fokko                             *
 *   fpbeekhof@gmail.com                                                   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be usefu,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You shoud have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#ifndef CVMLCPP_MODEM_H
#define CVMLCPP_MODEM_H 1

#include <bitset>
#include <limits>
#include <climits>
#include <exception>

#include <boost/static_assert.hpp>
#include <boost/integer/static_log2.hpp>

#include <omptl/omptl_numeric>

#include <cvmlcpp/base/stl_cmath.h>
#include <cvmlcpp/math/Vectors>
#include <cvmlcpp/signal/Channel>

namespace cvmlcpp
{

class ChannelUnknownOutputException: public std::exception
{
	virtual const char* what() const throw()
	{
		return "CVMLCPP::ChannelException: Channel class doesn't recognize output, gives zero probability for all possible inputs.";
	}
};
class ChannelInvalidResponseException: public std::exception
{
	virtual const char* what() const throw()
	{
		return "ChannelException: Channel returned invalid probability";
	}
};

template <typename T, std::size_t BITS>
T grayEncode(const T b)
{
	T res = 0;
	for (std::size_t i = 0; i < BITS; ++i)
	{
		const T blkSize = 1 << i;
		const T value =
			( (b >= blkSize) &&
			  ( (((b-blkSize)/(2*blkSize)) % 2) == 0u) ) ? 1 : 0;
		res |= value << i;
	}

	return res;
}


template <typename T, std::size_t BITS>
T grayDecode(const T n)
{
	bool invert = false;

	T res = 0;
	for (long int i = BITS-1; i >= 0; --i)
	{
		if (invert)
			res |= (~n & (1 << i));
		else
			res |= (n & (1 << i));
		invert = res & (1 << i); // bit set ?
	}

	return res;
}

template <typename T, std::size_t BITS>
void expand(const std::size_t x, std::tr1::array<T, BITS> &gray)
{
	for (std::size_t i = 0; i < BITS; ++i)
		gray[i] = (x & (1u << i)) ? T(1) : T(0);
}

template <typename T, std::size_t BITS>
void grayEncode(const std::size_t x, std::tr1::array<T, BITS> &gray)
{
	assert(BITS < CHAR_BIT*sizeof(std::size_t));
	assert(x < (1u << BITS));
	const std::size_t gx = grayEncode<std::size_t, BITS>(x);

	for (std::size_t i = 0; i < BITS; ++i)
		gray[i] = (gx & (1u << i)) ? T(1) : T(0);
}

template <typename T>
T _cdfNorm(const T x, const T mu, const T sigma)
{
	return 0.5 * (1.0 + std::tr1::erf( (x-mu) / (std::sqrt(2.0)*sigma) ) );
}

/* De-Modulate */
template <typename Input, typename Output, std::size_t N_SYMBOLS>
std::tr1::array<double, boost::static_log2<N_SYMBOLS>::value>
grayDeModulate(const double value,
		const std::tr1::array<Input, N_SYMBOLS> &symbols,
		const std::tr1::shared_ptr<Channel<Input, Output> > &channel)
{
	BOOST_STATIC_ASSERT(1<<boost::static_log2<N_SYMBOLS>::value==N_SYMBOLS);
	BOOST_STATIC_ASSERT(boost::integer_traits<Input>::is_integral);

	const std::size_t BITS = boost::static_log2<N_SYMBOLS>::value;

#ifndef WORD_BIT
	const std::size_t WORD_BIT = CHAR_BIT*sizeof(int);
#endif

	// Conditional probabilities p(y|x)
	cvmlcpp::StaticVector<double, N_SYMBOLS> p;
	for (std::size_t i = 0u; i < N_SYMBOLS; ++i)
	{
		const Input symbol = symbols[grayDecode<std::size_t, BITS>(i)];
		p[i] = channel->probabilityOf(value, symbol);
		if (! (p[i] >= 0.0) )
			throw ChannelInvalidResponseException();
		assert(p[i] >= 0.0);
		//assert(p[i] <= 1.0); // Not true for unquantified continuous distributions
	}
	// Scale such that sum of probablities is 1, thus transforming
	// the p(y|x) into p(x|y) by bayes' theorem.
	const double mass = std::accumulate(p.begin(), p.end(), 0.0);
	if (mass > 0.0)
		p *= (1.0 / mass);
	else
	{
		throw ChannelUnknownOutputException();
//		p = 0.0;
//		p[0] = 1.0; // Take a wild but valid guess
	}
	assert(std::abs(std::accumulate(p.begin(),p.end(),0.0) - 1.0) < 0.0001);

	// Calculate probabilities Pr[b=1] for each bit as
	// sum Pr[b=1|x_i] * Pr[X=x_i] where Pr[X=x_i] = p(x|y)
	cvmlcpp::StaticVector<double, BITS> quantization = 0.0;
	for (std::size_t i = 0u; i < N_SYMBOLS; ++i)
	{
		assert(p[i] >= 0.0);
		assert(p[i] <= 1.0);

		cvmlcpp::StaticVector<double, BITS> index; //gx;
//		grayEncode(i, gx);
		expand(i, index);
		quantization += p[i] * index; //gx;
	}

	for (std::size_t i = 0u; i < quantization.size(); ++i)
	{
		quantization[i] = cvmlcpp::clamp(quantization[i], 0.0, 1.0);
		assert(quantization[i] >= 0.0);
		assert(quantization[i] <= 1.0);
	}

	return quantization;
}

/* Modulate */
template <typename Input, std::size_t N_SYMBOLS>
Input grayModulate(const std::size_t message,
		   const std::tr1::array<Input, N_SYMBOLS> &symbols)
{
	BOOST_STATIC_ASSERT(boost::integer_traits<Input>::is_integral);
	BOOST_STATIC_ASSERT(N_SYMBOLS ==
			   (1u << boost::static_log2<N_SYMBOLS>::value) );
	const std::size_t BITS = boost::static_log2<N_SYMBOLS>::value;
	return symbols[grayDecode<std::size_t, BITS>(message)];
}

template <typename Input, std::size_t N_SYMBOLS>
class GrayModulator
{
	public:
		GrayModulator(const std::tr1::array<Input, N_SYMBOLS> &symbols):
			_symbols(symbols) { }

		Input operator()(const std::size_t chunk) const
		{
			assert(chunk < N_SYMBOLS);
			return grayModulate(chunk, _symbols);
		}

	private:
		const std::tr1::array<Input, N_SYMBOLS> &_symbols;
};

template <typename Input, typename Output, std::size_t N_SYMBOLS>
class GrayDemodulator
{
	public:
		static const std::size_t P = boost::static_log2<N_SYMBOLS>::value;

		typedef std::tr1::array<double, P> Quantization;

		typedef std::tr1::array<Input, N_SYMBOLS> InputArray;
		typedef std::tr1::shared_ptr<Channel<Input,Output> >ChannelPtr;

		GrayDemodulator(const InputArray &symbols,
				const ChannelPtr &channel) :
					symbols_(symbols), channel_(channel) { }

		Quantization operator()(const Output value) const
		{
			return grayDeModulate(value, symbols_, channel_);
		}

	private:
		const InputArray &symbols_;
		const ChannelPtr channel_;
};

} // end namespace

#endif
