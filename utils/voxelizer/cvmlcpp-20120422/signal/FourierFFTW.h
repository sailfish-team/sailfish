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

#ifndef CVMLCPP_FOURIER_FFTW
#define CVMLCPP_FOURIER_FFTW 1

#include <fftw3.h>

namespace cvmlcpp
{

namespace detail
{

/*
 *  Required: FFTPlan & FFTLib
 */

template <typename T>
struct FFTTraits { };
template <>
struct FFTTraits<double>
{
	typedef fftw_plan plan_type; 
	static bool init_threads() { return fftw_init_threads(); }
	static void execute(plan_type plan) { fftw_execute(plan); }
	static void destroy_plan(plan_type plan) { fftw_destroy_plan(plan); }
	static void plan_with_nthreads(const int threads) { fftw_plan_with_nthreads(threads); }
	static plan_type plan_dft    (int rank, const int *n, std::complex<double> *in, std::complex<double> *out, int sign, unsigned flags)
	{ return fftw_plan_dft    (rank, n, reinterpret_cast<fftw_complex *>(in), reinterpret_cast<fftw_complex *>(out), sign, flags); }
	static plan_type plan_dft_r2c(int rank, const int *n, double *in, std::complex<double> *out, unsigned flags)
	{ return fftw_plan_dft_r2c(rank, n, in, reinterpret_cast<fftw_complex *>(out), flags); }
	static plan_type plan_dft_c2r(int rank, const int *n, std::complex<double> *in, double *out, unsigned flags)
	{ return fftw_plan_dft_c2r(rank, n, reinterpret_cast<fftw_complex *>(in), out, flags); }
};
template <> struct FFTTraits<float>
{
	typedef fftwf_plan plan_type; 
	static bool init_threads() { return fftwf_init_threads(); }
	static void execute(plan_type plan) { fftwf_execute(plan); }
	static void destroy_plan(plan_type plan) { fftwf_destroy_plan(plan); }
	static void plan_with_nthreads(const int threads) { fftwf_plan_with_nthreads(threads); }
	static plan_type plan_dft    (int rank, const int *n, std::complex<float> *in, std::complex<float> *out, int sign, unsigned flags)
	{ return fftwf_plan_dft    (rank, n, reinterpret_cast<fftwf_complex *>(in), reinterpret_cast<fftwf_complex *>(out), sign, flags); }
	static plan_type plan_dft_r2c(int rank, const int *n, float *in, std::complex<float> *out, unsigned flags)
	{ return fftwf_plan_dft_r2c(rank, n, in, reinterpret_cast<fftwf_complex *>(out), flags); }
	static plan_type plan_dft_c2r(int rank, const int *n, std::complex<float> *in, float *out, unsigned flags)
	{ return fftwf_plan_dft_c2r(rank, n, reinterpret_cast<fftwf_complex *>(in), out, flags); }
};
template <> struct FFTTraits<long double>
{
	typedef fftwl_plan plan_type; 
	static bool init_threads() { return fftwl_init_threads(); }
	static void execute(plan_type plan) { fftwl_execute(plan); }
	static void destroy_plan(plan_type plan) { fftwl_destroy_plan(plan); }
	static void plan_with_nthreads(const int threads) { fftwl_plan_with_nthreads(threads); }
	static plan_type plan_dft    (int rank, const int *n, std::complex<long double> *in, std::complex<long double> *out, const int sign, const unsigned flags)
	{ return fftwl_plan_dft    (rank, n, reinterpret_cast<fftwl_complex *>(in), reinterpret_cast<fftwl_complex *>(out), sign, flags); }
	static plan_type plan_dft_r2c(int rank, const int *n, long double *in, std::complex<long double> *out, const unsigned flags)
	{ return fftwl_plan_dft_r2c(rank, n, in, reinterpret_cast<fftwl_complex *>(out), flags); }
	static plan_type plan_dft_c2r(int rank, const int *n, std::complex<long double> *in, long double *out, const unsigned flags)
	{ return fftwl_plan_dft_c2r(rank, n, reinterpret_cast<fftwl_complex *>(in), out, flags); }
};

template <typename T>
struct FFTPlan
{
	FFTPlan() : plan(0) { }
	typedef typename FFTTraits<T>::plan_type  plan_type;
	plan_type plan;
	bool ok() const { return plan != 0; }
};


template <typename T>
struct FFTLib
{
	typedef FFTPlan<T> plan_type;
	static bool init()
	{
		#ifdef USE_THREADS
			static bool initialized = false;

			#ifdef _OPENMP
			#pragma omp critical(CVMLCPP_FFT)
			#endif
			{
				if (!initialized)
					initialized = 
					#ifdef USE_THREADS
						FFTTraits<T>::init_threads() ? true : false;
					#else
						true;
					#endif
			}

			return initialized;
		#else
			return true;
		#endif
	}
	
	static bool execute(FFTPlan<T> &plan)
	{
		if (!plan.ok())
			return false;
		FFTTraits<T>::execute(plan.plan);
		return true;
	}
/*
	static bool loadWisdom(std::string wisdom)
	{
		FILE *input_file = std::fopen(wisdom.c_str(), "r");
		bool ok = (input_file != 0);
		if (ok)
		{
			#ifdef _OPENMP
			#pragma omp critical(CVMLCPP_FFT)
			#endif
			{
				ok = fftw_import_wisdom_from_file(input_file) ? true : false;
			}
			std::fclose(input_file);
		}
		return ok;
	}

	static void saveWisdom(std::string wisdom)
	{
		FILE *output_file = std::fopen(wisdom.c_str(), "w");
		if (output_file)
		{
			#ifdef _OPENMP
			#pragma omp critical(CVMLCPP_FFT)
			#endif
			{
				fftw_export_wisdom_to_file(output_file);
			}
			std::fclose(output_file);
		}
	}

	static void forgetWisdom()
	{
		#ifdef _OPENMP
		#pragma omp critical(CVMLCPP_FFT)
		#endif
		{
			fftw_forget_wisdom();
		}
	}
*/
	static void destroyPlan(FFTPlan<T> &plan)
	{
		#ifdef _OPENMP
		#pragma omp critical(CVMLCPP_FFT)
		#endif
		{
			if (plan.ok())
				FFTTraits<T>::destroy_plan(plan.plan);
		}
	}

	static void plan_with_nthreads(const int nthreads)
	{
		#ifdef USE_THREADS
			#ifdef _OPENMP
			#pragma omp critical(CVMLCPP_FFT)
			#endif
			{
				FFTTraits<T>::plan_with_nthreads(nthreads);
			}
		#endif
	}

	static plan_type plan_dft    (int rank, const int *n, std::complex<T> *in, std::complex<T> *out, const bool forward, const unsigned flags)
	{
		FFTPlan<T> plan;
		const int sign = forward ? FFTW_FORWARD : FFTW_BACKWARD;
		plan.plan = FFTTraits<T>::plan_dft(rank, n, in, out, sign, flags);
		return plan;
	}
	static plan_type plan_dft_r2c(int rank, const int *n, T *in, std::complex<T> *out, const unsigned flags)
	{
		FFTPlan<T> plan;
		plan.plan = FFTTraits<T>::plan_dft_r2c(rank, n, in, out, flags);
		return plan;
	}
	static plan_type plan_dft_c2r(int rank, const int *n, std::complex<T> *in, T *out, const unsigned flags)
	{
		FFTPlan<T> plan;
		plan.plan = FFTTraits<T>::plan_dft_c2r(rank, n, in, out, flags);
		return plan;
	}
};


} // namespace detail

} // namespace cvmlcpp

#endif
