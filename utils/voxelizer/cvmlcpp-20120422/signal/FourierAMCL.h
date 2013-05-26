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

#ifndef CVMLCPP_FOURIER_ACML
#define CVMLCPP_FOURIER_ACML 1

// 
// Thanks, AMD!!
// This is legal as std::complex<T> is supposedly binary compatible with T[2].
//
#define _ACML_COMPLEX
typedef std::complex<float> complex;
typedef std::complex<double> doublecomplex;

#include <acml.h>

namespace cvmlcpp
{

namespace detail
{

template <class Arrayish>
void to_FFTW_format(Arrayish &a, const std::size_t N)
{
	typename Arrayish::value_type tmp = a[N-1]; // for dest = 1
	for (std::size_t src = 1, dest = 2; dest != 1; 
	     dest = (src < N/2) ? 2ul*src : 2ul*(N-src)-1ul)
	{
		swap(tmp, src);
		swap(tmp, dest);
	}
}



// Required 
template <typename T>
struct FFTPlan
{
	FFTPlan() : plan(0) { }
	typedef typename FFTWTraits<T>::plan_type  plan_type;
	plan_type plan;
	bool ok() const { return plan != 0; }
};

template <typename T>
struct FFTLibCalls
{
	static bool init()
	{
		return true;
	}

	static bool execute(FFTPlan<T> &plan)
	{
		if (!plan.ok())
			return false;
		Traits<T>::execute(plan.plan);
		return true;
	}

	static void destroyPlan(FFTPlan<T> &plan) { }

	static void plan_with_nthreads(const int threads) { }
	
	static typename Traits<T>::plan_type plan_dft
	(int rank, const int *n, std::complex<T> *in, std::complex<T> *out, int sign, unsigned flags)
	{ return Traits<T>::plan_dft    (rank, n, in, out, sign, flags); }
	
	static typename Traits<T>::plan_type plan_dft_r2c
	(int rank, const int *n, T *in, std::complex<T> *out, unsigned flags)
	{ return Traits<T>::plan_dft_r2c(rank, n, in, out, flags); }
	
	static typename Traits<T>::plan_type plan_dft_c2r
	(int rank, const int *n, std::complex<T> *in, T *out, unsigned flags)
	{ return Traits<T>::plan_dft_c2r(rank, n, in, out, flags); }
	
};

template <typename T>
class Planner
{
	public:
	template <
	template <typename Tm, std::size_t D, typename Aux> class ArrayIn,
	template <typename Tm, std::size_t D, typename Aux> class ArrayOut,
		std::size_t N, typename Tin, typename Tout,
		typename AuxIn, typename AuxOut>
	static FFTPlan<T>
	makeDFTPlan(ArrayIn<Tin, N, AuxIn> &in, ArrayOut<Tout, N, AuxOut> &out,
		    bool forward, unsigned flags, unsigned threads)
	{
		typedef typename ValueType<Tin> ::value_type VTin;
		typedef typename ValueType<Tout>::value_type VTout;

		#if defined _OPENMP && defined USE_THREADS
		threads = std::max(threads,
			static_cast<unsigned>(omp_get_max_threads()));
		#endif

		if (threads < 1u)
			threads = 1u;
		FFTLib<T>::init();

		FFTLib<T>::plan_with_nthreads(threads);
		FFTPlan<T> plan;
		#ifdef _OPENMP
		#pragma omp critical(CVMLCPP_FFT)
		#endif
		{
			plan.plan = makeDFTPlan_(in, out, forward, flags);
		}

		return plan;
	}

	private:
	// Any dimension; complex --> complex
	template <
	template <typename Tm, std::size_t D, typename Aux> class ArrayIn,
	template <typename Tm, std::size_t D, typename Aux> class ArrayOut,
		std::size_t N, typename AuxIn, typename AuxOut>
	static typename FFTPlan<T>::plan_type
	makeDFTPlan_(ArrayIn<std::complex<T>, N, AuxIn> &in,
		     ArrayOut<std::complex<T>, N, AuxOut> &out,
			bool forward, unsigned flags)
	{
		typedef array_traits<ArrayIn,  std::complex<T>, N, AuxIn> at1;
		typedef array_traits<ArrayOut, std::complex<T>, N, AuxOut>at2;

		int sign = forward ? FFTW_FORWARD : FFTW_BACKWARD;
		int dims[N];
		std::copy(at1::shape(in), at1::shape(in)+N, dims);

		// Resize output
		at2::resize(out, dims);

		return FFTWTraits<T>::plan_dft(N, dims, &(*at1::begin(in)), &(*at2::begin(out)),
//			reinterpret_cast<complex<T> *>(&(*at1::begin(in))),
//			reinterpret_cast<complex<T> *>(&(*at2::begin(out))),
				sign, flags);
	}

	// Any dimension; real --> complex
	template <
	template <typename Tm, std::size_t D, typename Aux> class ArrayIn,
	template <typename Tm, std::size_t D, typename Aux> class ArrayOut,
		std::size_t N, typename AuxIn, typename AuxOut>
	static typename FFTPlan<T>::plan_type
	makeDFTPlan_(ArrayIn<T, N, AuxIn> &in,
				ArrayOut<std::complex<T>, N, AuxOut> &out,
				bool forward, unsigned flags)
	{
		assert(forward); // User input inconsistent ?
		typedef array_traits<ArrayIn,  T, N, AuxIn> at1;
		typedef array_traits<ArrayOut, std::complex<T>, N, AuxOut>at2;

		// Last dimension must be even!
		if (at1::shape(in)[N-1] % 2)
			return 0;

		int dimsIn[N], dimsOut[N];
		std::copy(at1::shape(in), at1::shape(in)+N, dimsIn);

		// Resize output
		std::copy(dimsIn, dimsIn+N, dimsOut);
		dimsOut[N-1] = 1 + dimsOut[N-1]/2; // Last dimension is special
		at2::resize(out, dimsOut);

		typename at2::iterator outBegin = at2::begin(out);

		return FFTWTraits<T>::plan_dft_r2c(N, dimsIn,
		    static_cast<T*>(&(*at1::begin(in))), &(*outBegin), flags);
	}

	// Any dimension; complex --> real
	template <
	template <typename Tm, std::size_t D, typename Aux> class ArrayIn,
	template <typename Tm, std::size_t D, typename Aux> class ArrayOut,
		std::size_t N, typename AuxIn, typename AuxOut>
	static typename FFTPlan<T>::plan_type
	makeDFTPlan_(ArrayIn<std::complex<T>, N, AuxIn> &in,
				ArrayOut<T, N, AuxOut> &out,
				bool forward, unsigned flags)
	{
		assert(!forward); // User input inconsistent ?
		typedef array_traits<ArrayIn,  std::complex<T>, N, AuxIn> at1;
		typedef array_traits<ArrayOut, T, N, AuxOut> at2;

		int dimsOut[N];
		std::copy(at1::shape(in), at1::shape(in)+N, dimsOut);
		dimsOut[N-1] = (dimsOut[N-1]-1)*2;

		// Resize output
		at2::resize(out, dimsOut);

		return FFTWTraits<T>::plan_dft_c2r(N, dimsOut,
			&(*at1::begin(in)),
//			reinterpret_cast<fftw_complex *>(&(*at1::begin(in))),
//			static_cast<VT2 *>(&(*at2::begin(out))), flags);
			&(*at2::begin(out)), flags);
	}
};

} // namespace detail

} // namespace cvmlcpp

#endif
