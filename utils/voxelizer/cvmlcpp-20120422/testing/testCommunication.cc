/***************************************************************************
 *   Copyright (C) 2009 by BEEKHOF, Fokko                                  *
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
#include <climits>
#include <tr1/array>

#include <boost/integer_traits.hpp>
#include <cvmlcpp/signal/Communication>

#include <cvmlcpp/math/Math>
#include <cvmlcpp/math/Vectors>
#include <cvmlcpp/base/StringTools>

int main()
{
	using namespace cvmlcpp;
	using namespace std;
	using namespace boost;

	typedef unsigned short Symbol;
	const size_t NSymb = 16u;
	const tr1::array<Symbol, NSymb> symbols =
		generateAlphabet<Symbol, NSymb>();

	MLLDPCCode<8, 4>::G g;
	g[0] = std::bitset<4>(std::string("1111"));
	g[1] = std::bitset<4>(std::string("1101"));
	g[2] = std::bitset<4>(std::string("1011"));
	g[3] = std::bitset<4>(std::string("0111"));
// 	BPLDPCCode<8, 4> code(g, 8u);
	MLLDPCCode<8, 4> code(g);

	std::bitset<4> m(std::string("1011"));
	std::bitset<8> cw;

	code.encode(m, cw);

// 	std::cout << "Message  " << m.to_string() << std::endl;

// 	std::cout << "Codeword " << cw.to_string() << std::endl;
	assert(cw == std::bitset<8>(std::string("10110101")));

	std::tr1::array<double, 8> co; // channel output
	for (unsigned i = 0u; i < 8u; ++i)
		co[i] = cw[i];

// 	std::cout << "Channel output " <<
// 		cvmlcpp::to_string(co.begin(), co.end()) << std::endl;

	std::bitset<4> m_est1;
	code.decode(co, m_est1);
// 	std::cout << "m_est1 " << m_est1.to_string() << std::endl;
	assert(m_est1 == m);

// 	for (unsigned i = 0u; i < 8u; ++i)
// 		co[i] = (cw[i] ? 0.9 : 0.1) + 0.01*float(i);

	co[0u] = 0.4; //1.0 - co[0u]; // one bit error
	std::bitset<4> m_est2;
	code.decode(co, m_est2);
// 	std::cout << "m_est2 " << m_est2.to_string() << std::endl;
	assert(m_est2 == m);

	const unsigned L = 2;
	const unsigned K = L*sizeof(unsigned short) * CHAR_BIT;
	const unsigned N = 4*K;
	typedef Communicator<N, K, Symbol, float, NSymb> MyComm;
	MyComm comm;

// 	comm.print();
	std::tr1::array<Symbol, MyComm::LENGTH> channelInput;
	std::tr1::array<unsigned short, L> mesg;

	for (unsigned i = 0u; i < L; ++i)
		mesg[i] = 32+i*32;
// 	std::cout << "Mesg: ";
// 	std::cout << cvmlcpp::to_string(mesg.begin(),
// 					mesg.end()) << std::endl;
	assert(comm.encode(mesg.begin(), channelInput.begin()));
	for (unsigned i = 0u; i < L; ++i)
		assert(mesg[i] == 32+i*32);

// 	std::cout << "Sent Codeword: ";
// 	std::cout << cvmlcpp::to_string(channelInput.begin(),
// 					channelInput.end()) << std::endl;

	std::tr1::array<float, MyComm::LENGTH> channelOutput;
	std::copy(channelInput.begin(), channelInput.end(),
		  channelOutput.begin());

	std::tr1::array<unsigned short, L> mesg_est;
	mesg_est[0] = 0; // wrong answer

	// same decoder
	assert(comm.decode(channelOutput.begin(), mesg_est.begin()));
	assert(mesg_est == mesg);
	for (unsigned i = 0u; i < L; ++i)
		assert(mesg_est[i] == mesg[i]);

	mesg_est[0] = 7; // wrong answer
	assert(comm.decode(channelOutput.begin(), mesg_est.begin()));
	assert(mesg_est == mesg);
// 	std::cout << "Mesg_EST: ";
// 	std::cout << cvmlcpp::to_string(mesg_est.begin(),
// 					mesg_est.end()) << std::endl;
	for (unsigned i = 0u; i < L; ++i)
		assert(mesg_est[i] == mesg[i]);

	// other decoder
	mesg_est[0] = 7; // wrong answer

// 	std::cout << "Noisy Codeword: ";
	std::tr1::array<float, MyComm::LENGTH> noise;
	for (unsigned i = 0; i < MyComm::LENGTH; ++i)
// 		noise[i] += 2048 + i*(1u<<(CHAR_BIT*sizeof(Symbol)/3u));
// 		noise[i] = 512 + i*(1u<<(CHAR_BIT*sizeof(Symbol)/3u));
		noise[i] = i*(1u<<(CHAR_BIT*sizeof(Symbol)/4u));

	MyComm commDec( MyComm::ChannelPtr(new AWGNChannel<Symbol, float>(
				 variance(noise.begin(), noise.end(), 0.0f) )));
	assert(commDec.decode(channelOutput.begin(), mesg_est.begin()));
// 	std::cout << "Mesg_EST: ";
// 	std::cout << cvmlcpp::to_string(mesg_est.begin(),
// 					mesg_est.end()) << std::endl;
	for (unsigned i = 0u; i < L; ++i)
		assert(mesg_est[i] == mesg[i]);

// 	std::cout << cvmlcpp::to_string(channelOutput.begin(),
// 					channelOutput.end()) << std::endl;
	std::transform(channelOutput.begin(), channelOutput.end(),
		noise.begin(), channelOutput.begin(), std::plus<float>());

	mesg_est[0] = 7; // wrong answer
	assert(commDec.decode(channelOutput.begin(), mesg_est.begin()));
	assert(mesg_est == mesg);


// 	std::cout << "MEst: ";
// 	std::cout << cvmlcpp::to_string(mesg_est.begin(),
// 					mesg_est.end());
// 	std::cout << " was: ";
// 	std::cout << cvmlcpp::to_string(mesg.begin(),
// 					mesg.end()) << std::endl;

	return 0;
}

