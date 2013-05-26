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

#ifndef CVMLCPP_CODE_H
#define CVMLCPP_CODE_H 1

#include <bitset>
#include <cassert>
#include <tr1/array>

#include <boost/static_assert.hpp>
#include <boost/integer_traits.hpp>

namespace cvmlcpp
{

// N: bits in codeword; K bits in message
template <std::size_t N, std::size_t K>
class Code
{
	public:
		static const std::size_t M = N-K; // Number of parity checks
		typedef std::tr1::array<std::bitset<K>, M> G;
		typedef std::tr1::array<std::bitset<N>, M> H;
		typedef std::tr1::array<double, N> Quantization;

		// G only needs to contain entries for parity checks,
		// i.e. copying the message will be done automatically.
		Code(const G &generator) : _g(generator)
		{
			BOOST_STATIC_ASSERT(N >= K);

			// Derive check matrix H from generator matrix G
			for (std::size_t chk = 0u; chk < M; ++chk) // nr check
			{
				_h[chk].reset();
				_h[chk].set(chk, true); // parity bit of check

				// nr mesg bit
				for (std::size_t m = 0; m < K; ++m)
					_h[chk].set(M+m, _g[chk].test(m));
			}
		}

		virtual ~Code() { }

		bool encode(const std::bitset<K> &message,
			    std::bitset<N> &codeword) const
		{
			// Add parity checks
			for (std::size_t i = 0u; i < M; ++i)
				codeword.set(i, (message&_g[i]).count()%2u);
			// copy message
			for (std::size_t i = 0u; i < K; ++i)
				codeword.set(M+i, message[i]);

			assert(verify(codeword));

			return true;
		}

		// Quantization is a list of probabilities; i.e.
		// for each bit, the probability that that bit is 1.
		bool decode(const Quantization &quantization,
			    std::bitset<K> &message)
		{
			std::bitset<N> codeword = directDecode(quantization);
// std::cout << "Code::decode()  direct CW: " << codeword.to_string() <<
// std::endl;
			bool ok = verify(codeword);
			if (!ok)
			{
// std::cout << "Code::decode() correcting..." << std::endl;
				ok = this->correct(quantization, codeword);
			}

			for (std::size_t i = 0u; i < K; ++i) // copy message
				message.set(i, codeword[M+i]);
// std::cout << "Code::decode() correct CW: " << codeword.to_string() <<
// std::endl;

			return ok;
		}

		void print() const
		{
			std::cout << "G:" <<std::endl;
			for (std::size_t chk = 0u; chk < M; ++chk)
				std::cout << _g[chk].to_string() <<std::endl;
			std::cout << "H:" <<std::endl;
			for (std::size_t chk = 0u; chk < M; ++chk)
				std::cout << _h[chk].to_string() <<std::endl;
		}

	protected:
		virtual bool correct(const
				Quantization &quantization,
				std::bitset<N> &codeword) = 0;

		bool verify(const std::bitset<N> &codeword) const
		{
// std::cout << "Code::verify() " << std::endl;
// for (std::size_t i = 0; i < M; ++i)
// 	std::cout << "CW " << codeword << std::endl << "H: " << _h[i] <<
// 		std::endl << "S: " << (codeword & _h[i]) << "   -->  " <<
// 		(codeword & _h[i]).count() << std::endl <<
// 		"############################################" << std::endl;

			for (std::size_t i = 0; i < M; ++i)
				if ( (codeword & _h[i]).count() % 2u )
					return false;
			return true;
		}

		const std::bitset<N> &h(const std::size_t chk_index) const
		{
			assert(chk_index < M);
			return _h[chk_index];
		}

		const std::tr1::array<std::bitset<N>, M> &h() const
		{ return _h; }

	private:
		G _g;
		H _h;

		static std::bitset<N> directDecode(
				const Quantization &quantization)
		{
			std::bitset<N> codeword;
			for (std::size_t i = 0; i < N; ++i)
				codeword.set(i, quantization[i] > 0.5);
			return codeword;
		}
};

} // namespace

#endif
