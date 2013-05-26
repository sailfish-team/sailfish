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

#ifndef CVMLCPP_FOURIER_CUFFT
#define CVMLCPP_FOURIER_CUFFT 1

#ifdef _OPENMP
#error "CUFFT and OpenMP are incompatible."
#endif

#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>

// FIXME: This should be a value provided by NVidia, but I can't find it
#define CVMLCPP_CUFFT_INVALID_CUFFTPLAN -1

#define FFTW_MEASURE 0
#define FFTW_ESTIMATE 1
#define FFTW_DESTROY_INPUT 0

namespace cvmlcpp
{

namespace detail
{

/*
 *  Required: FFTPlan & FFTLib
 */

template <typename T>
struct FFTPlan
{
	typedef cufftHandle plan_type;

	FFTPlan() : plan(CVMLCPP_CUFFT_INVALID_CUFFTPLAN),
			in(0), out(0), cu_in(0), cu_out(0),
			mem_size_in(0), mem_size_out(0), sign(0),
			status(CUFFT_INVALID_PLAN), stream(0)/*, context(0) */{ }

	plan_type plan;
	void *in, *out; // Main Memory
	void *cu_in, *cu_out; // Cuda memory on the device
	std::size_t mem_size_in, mem_size_out;
	int sign; // C2C only: Forward or Backward transform ?
	cufftType type;
	bool ok() const { return CUFFT_SUCCESS == status; }
	cufftResult status;
	cudaStream_t stream;
//	CUcontext * context;
};

template <typename T>
struct FFTTraits {};

template <>
struct FFTTraits<cufftReal>
{
	static const cufftType C2C = CUFFT_C2C;
	static const cufftType R2C = CUFFT_R2C;
	static const cufftType C2R = CUFFT_C2R;

	static bool cufftExec( const FFTPlan<cufftReal> &plan )
	{
		if (!plan.ok())
		{
			std::cerr << "FFTTraits<cufftReal>: plan not OK" << std::endl;
			return false;
		}
		cufftResult result = CUFFT_INVALID_PLAN;

		if ( plan.cu_in != plan.in &&
//		     cudaMemcpyAsync(plan.cu_in, plan.in, plan.mem_size_in, cudaMemcpyHostToDevice, plan.stream) != cudaSuccess)
		     cudaMemcpy(plan.cu_in, plan.in, plan.mem_size_in, cudaMemcpyHostToDevice) != cudaSuccess)
		{
			std::cerr << "FFTTraits<cufftReal>: cudaMemcpy in failed" << std::endl;
			return false;
		}
		cudaStreamSynchronize(plan.stream);
		switch(plan.type)
		{
			case C2C: result = cufftExecC2C(plan.plan, (cufftComplex *)plan.cu_in, (cufftComplex *)plan.cu_out, plan.sign);
				  break;
			case R2C: result = cufftExecR2C(plan.plan, (cufftReal *)plan.cu_in, (cufftComplex *)plan.cu_out);
				  break;
			case C2R: result = cufftExecC2R(plan.plan, (cufftComplex *)plan.cu_in, (cufftReal *)plan.cu_out);
				  break;
			default: assert(false); return false;
		}
		cudaStreamSynchronize(plan.stream);
		bool ok = (result == CUFFT_SUCCESS);
		if (!ok) std::cerr << "FFTTraits<cufftReal>: cufftExec failed" << std::endl;
		if (ok)
		{
			if ( plan.cu_out != plan.out &&
//			     cudaMemcpyAsync(plan.out, plan.cu_out, plan.mem_size_out, cudaMemcpyDeviceToHost, plan.stream) != cudaSuccess)
			     cudaMemcpy(plan.out, plan.cu_out, plan.mem_size_out, cudaMemcpyDeviceToHost) != cudaSuccess)
				ok = false;
			if (!ok) std::cerr << "FFTTraits<cufftReal>: cudaMemcpy out failed" << std::endl;
			cudaStreamSynchronize(plan.stream);
		}
		return ok;
	}
};

template <>
struct FFTTraits<cufftDoubleReal>
{

	static const cufftType C2C = CUFFT_Z2Z;
	static const cufftType R2C = CUFFT_D2Z;
	static const cufftType C2R = CUFFT_Z2D;

	static bool cufftExec( const FFTPlan<cufftDoubleReal> &plan )
	{
		if (!plan.ok())
		{
			std::cerr << "FFTTraits<cufftReal>: plan not OK" << std::endl;
			return false;
		}
		cufftResult result = CUFFT_INVALID_PLAN;

		if ( plan.cu_in != plan.in &&
//		     cudaMemcpyAsync(plan.cu_in, plan.in, plan.mem_size_in, cudaMemcpyHostToDevice, plan.stream) != cudaSuccess)
		     cudaMemcpy(plan.cu_in, plan.in, plan.mem_size_in, cudaMemcpyHostToDevice) != cudaSuccess)
		{
			std::cerr << "FFTTraits<cufftDoubleReal>: cudaMemcpy cu_in failed" << std::endl;
			return false;
		}
		cudaStreamSynchronize(plan.stream);
		switch(plan.type)
		{
			case C2C: result = cufftExecZ2Z(plan.plan, (cufftDoubleComplex *)plan.cu_in, (cufftDoubleComplex *)plan.cu_out, plan.sign);
				  break;
			case R2C: result = cufftExecD2Z(plan.plan, (cufftDoubleReal *)plan.cu_in, (cufftDoubleComplex *)plan.cu_out);
				  break;
			case C2R: result = cufftExecZ2D(plan.plan, (cufftDoubleComplex *)plan.cu_in, (cufftDoubleReal *)plan.cu_out);
				  break;
			default: assert(false); return false;
		}
		cudaStreamSynchronize(plan.stream);
		bool ok = (result == CUFFT_SUCCESS);
		if (!ok) std::cerr << "FFTTraits<cufftDoubleReal>: cufftExec failed" << std::endl;
		if (ok)
		{
			if ( plan.cu_out != plan.out &&
//			     cudaMemcpyAsync(plan.out, plan.cu_out, plan.mem_size_out, cudaMemcpyDeviceToHost, plan.stream) != cudaSuccess)
			     cudaMemcpy(plan.out, plan.cu_out, plan.mem_size_out, cudaMemcpyDeviceToHost) != cudaSuccess)
				ok = false;
			if (!ok) std::cerr << "FFTTraits<cufftDoubleReal>: cudaMemcpy cu_out failed" << std::endl;
			cudaStreamSynchronize(plan.stream);
		}
		return ok;
	}
};

template <typename T>
struct CUAllocator
{
	static bool alloc(const int rank, const int *n, const std::complex<T> * const in,
			  const std::complex<T> * const out, FFTPlan<T> &plan)
	{
		assert(rank >= 1 && rank <= 3);
		plan.mem_size_in = sizeof(std::complex<T>);
		for (int i = 0; i < rank; ++i)
			plan.mem_size_in *= n[i];
		plan.mem_size_out = plan.mem_size_in;
		return alloc(plan);
	}

	static bool alloc(const int rank, const int *n, const T * const in,
			  const std::complex<T> * const out, FFTPlan<T> &plan)
	{
		assert(rank >= 1 && rank <= 3);
		plan.mem_size_in = sizeof(T);
		for (int i = 0; i < rank; ++i)
			plan.mem_size_in *= n[i];
		plan.mem_size_out = sizeof(std::complex<T>);
		for (int i = 0; i < rank-1; ++i)
			plan.mem_size_out *= n[i];
		plan.mem_size_out *= 1 + n[rank-1] / 2;
		return alloc(plan);
	}

	static bool alloc(const int rank, const int *n, const std::complex<T> * const in,
			  const T * const out, FFTPlan<T> &plan)
	{
		assert(rank >= 1 && rank <= 3);
		plan.mem_size_in = sizeof(std::complex<T>);
		for (int i = 0; i < rank-1; ++i)
			plan.mem_size_in *= n[i];
		plan.mem_size_in *= 1 + n[rank-1] / 2;
		plan.mem_size_out = sizeof(T);
		for (int i = 0; i < rank; ++i)
			plan.mem_size_out *= n[i];

		return alloc(plan);
	}

	static bool alloc(FFTPlan<T> &plan)
	{
//std::cerr << "alloc(): size_in: " << plan.mem_size_in << " size_out: " << plan.mem_size_out << std::endl;
		if (cudaMalloc(&plan.cu_in,  plan.mem_size_in) != cudaSuccess)
		{
			std::cerr << "CUAllocator: cudaMalloc mem_size_in failed (" << plan.mem_size_in << ")" << std::endl;
			return false;
		}
		if (cudaMalloc(&plan.cu_out, plan.mem_size_out) != cudaSuccess)
		{
			std::cerr << "CUAllocator: cudaMalloc mem_size_out failed (" << plan.mem_size_out << ")" << std::endl;
			cudaFree(plan.cu_in);
			return false;
		}
		return true;
	}
};

template <typename T>
struct FFTPlanner
{
	template <typename U, typename V>
	static FFTPlan<T> cufft_make_plan(const int rank, const int *n, U *in, V *out, const cufftType type, const int sign = 0)
	{
		FFTPlan<T> plan;

		plan.in   = reinterpret_cast<void *>(in);
		plan.out  = reinterpret_cast<void *>(out);
		plan.type = type;
		plan.sign = sign;
/*
This is not it...
#ifdef _OPENMP
		CUdevice device;
		uCtxGetDevice(&device);
		if (cuCtxCreate(&plan.context, CU_CTX_SCHED_YIELD, device) != CUDA_SUCCESS)//cudaSuccess)
		{
			std::cerr << "FFTPlanner: cuCtxCreate failed" << std::endl;
			return plan;
		}
#endif
 */

/*
std::cerr << "FFTPlanner: sizeof(T) " << sizeof(T) << " rank " << rank << " transformtype " << type << " size ";
for (int i = 0; i < rank; ++i)
	std::cerr << n[i] << " ";
std::cerr << std::endl;
*/
		if (!CUAllocator<T>::alloc(rank, n, in, out, plan))
		{
			std::cerr << "FFTPlanner: alloc failed" << std::endl;
			return plan;
		}

		switch(rank)
		{
			case 1: plan.status = cufftPlan1d( &plan.plan, n[0], type, 1 );
				break;
			case 2: plan.status = cufftPlan2d( &plan.plan, n[0], n[1], type );
				break;
			case 3: plan.status = cufftPlan3d( &plan.plan, n[0], n[1], n[2], type );
				break;
			default: assert(false); return plan;
		}

		if (plan.status != CUFFT_SUCCESS)
		{
			std::cerr << "FFTPlanner: cufftPlan failed: " << plan.status << std::endl;
			plan.plan = CVMLCPP_CUFFT_INVALID_CUFFTPLAN;
			cudaFree(plan.cu_in);
			cudaFree(plan.cu_out);
			return plan;
		}

		// Create proper stream for plan
		if (cudaStreamCreate(&plan.stream) != cudaSuccess)
		{
			std::cerr << "FFTPlanner: cudaStreamCreate failed" << std::endl;
			cufftDestroy(plan.plan);
			plan.plan = CVMLCPP_CUFFT_INVALID_CUFFTPLAN;
			cudaFree(plan.cu_in);
			cudaFree(plan.cu_out);
			return plan;
		}
		cufftSetStream(plan.plan, plan.stream);

		// Set FFTW memory layout compatibility mode
		plan.status = cufftSetCompatibilityMode( plan.plan, CUFFT_COMPATIBILITY_FFTW_ALL );
		if (plan.status != CUFFT_SUCCESS)
		{
			std::cerr << "FFTPlanner: cufftSetCompatibilityMode failed" << std::endl;
			cufftDestroy(plan.plan);
			plan.plan = CVMLCPP_CUFFT_INVALID_CUFFTPLAN;
			cudaFree(plan.cu_in);
			cudaFree(plan.cu_out);
			cudaStreamDestroy(plan.stream);
			return plan;
		}

		std::cerr << "FFTPlanner: plan made: " << plan.plan << std::endl;
		return plan;
	}
#ifdef USE_THRUST
	template <typename U, typename V>
	static FFTPlan<T> cufft_make_gpu_plan(const int rank, const int *n, U *in, V *out, const cufftType type, const int sign = 0)
	{
//std::cerr << "cufft_make_gpu_plan" << std::endl;
		FFTPlan<T> plan;

		plan.in   = plan.cu_in   = reinterpret_cast<void *>(in);
		plan.out  = plan.cu_out  = reinterpret_cast<void *>(out);
		plan.type = type;
		plan.sign = sign;

		switch(rank)
		{
			case 1: plan.status = cufftPlan1d( &plan.plan, n[0], type, 1 );
				break;
			case 2: plan.status = cufftPlan2d( &plan.plan, n[0], n[1], type );
				break;
			case 3: plan.status = cufftPlan3d( &plan.plan, n[0], n[1], n[2], type );
				break;
			default: assert(false); return plan;
		}

		if (plan.status != CUFFT_SUCCESS)
		{
			std::cerr << "FFTPlanner: cufftPlan failed" << plan.status << std::endl;
			plan.plan = CVMLCPP_CUFFT_INVALID_CUFFTPLAN;
			return plan;
		}

		cudaThreadSynchronize();
		plan.status = cufftSetCompatibilityMode( plan.plan, CUFFT_COMPATIBILITY_FFTW_ALL );
		if (plan.status != CUFFT_SUCCESS)
		{
			std::cerr << "FFTPlanner: cufftSetCompatibilityMode failed" << std::endl;
			cufftDestroy(plan.plan);
			plan.plan = CVMLCPP_CUFFT_INVALID_CUFFTPLAN;
			return plan;
		}

		return plan;
	}
#endif
};

template <typename T>
struct FFTLib
{
	typedef FFTPlan<T> plan_type;
	static bool init() { return true; }

	static bool execute(FFTPlan<T> &plan)
	{
		if (!plan.ok())
		{
			std::cerr << "FFTLib::execute: plan not ok" << std::endl;
			return false;
		}
		const bool ok = FFTTraits<T>::cufftExec(plan);
		if (!ok) std::cerr << "FFTLib::execute: cufftExec failed" << std::endl;
		return ok;
//		return FFTTraits<T>::cufftExec(plan);
	}

	static void destroyPlan(FFTPlan<T> &plan)
	{
		if (plan.ok())
		{
			assert(plan.plan != CVMLCPP_CUFFT_INVALID_CUFFTPLAN);
			const cufftResult result = cufftDestroy(plan.plan);
			if (result != CUFFT_SUCCESS)
				std::cerr << "cufftDestroy() returned " << result << std::endl;
			assert(result != CUFFT_INVALID_PLAN);
			cudaFree(plan.cu_in );
			cudaFree(plan.cu_out);
			cudaStreamDestroy(plan.stream);
			std::cerr << "FFTLib: plan destroyed: " << plan.plan << std::endl;
			plan = FFTPlan<T>(); // Re-initialize
		}
		else
			std::cerr << "destroyPlan(): plan: " << plan.plan << " not ok, not destroyed." << std::endl;
		assert(!plan.ok());
	}

	static void plan_with_nthreads(const int threads) { }

	static plan_type plan_dft(const int rank, const int *n, std::complex<T> *in, std::complex<T> *out, const bool forward, const unsigned flags)
	{
		const int sign = forward ? CUFFT_FORWARD : CUFFT_INVERSE;
		return FFTPlanner<T>::cufft_make_plan(rank, n, in, out, FFTTraits<T>::C2C, sign);
	}

//	template <typename It1, typename It2>
//	static plan_type plan_dft_r2c(const int rank, const int *n, const It1 in, const It2 out, const unsigned flags)
//	{ return FFTPlanner<T>::cufft_make_plan(rank, n, &(*in), &(*out), FFTTraits<T>::R2C); }
	static plan_type plan_dft_r2c(const int rank, const int *n, T *in, std::complex<T> *out, const unsigned flags)
	{ return FFTPlanner<T>::cufft_make_plan(rank, n, in, out, FFTTraits<T>::R2C); }

#ifdef USE_THRUST
	static plan_type plan_dft_r2c(const int rank, const int *n,
		thrust::device_ptr<T> in, thrust::device_ptr<std::complex<T> > *out,
		const unsigned flags)
	{ return FFTPlanner<T>::cufft_make_gpu_plan(rank, n, in, out, FFTTraits<T>::R2C); }
#endif

	static plan_type plan_dft_c2r(const int rank, const int *n, std::complex<T> *in, T *out, const unsigned flags)
	{ return FFTPlanner<T>::cufft_make_plan(rank, n, in, out, FFTTraits<T>::C2R); }

#ifdef USE_THRUST
	static plan_type plan_dft_c2r(const int rank, const int *n,
		thrust::device_ptr<std::complex<T> > in, thrust::device_ptr<T> *out,
		const unsigned flags)
	{ return FFTPlanner<T>::cufft_make_gpu_plan(rank, n, in, out, FFTTraits<T>::C2R); }
#endif

};

} // namespace detail

} // namespace cvmlcpp

#endif
