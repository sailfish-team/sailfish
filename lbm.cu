// the following additional constants need to be defined:
// LAT_H, LAT_W, BLOCK_SIZE, GEO_FLUID, GEO_WALL, GEO_INFLOW

#define DT 1.0f

__constant__ float tau;			// relaxation time

struct DistP {
	float *fC, *fE, *fW, *fS, *fN, *fSE, *fSW, *fNE, *fNW;
};

struct Dist {
	float fC, fE, fW, fS, fN, fSE, fSW, fNE, fNW;
};

//
// Copy the idx-th distributin from din into dout.
//
__device__ void inline getDist(Dist &dout, DistP din, int idx)
{
	dout.fC = din.fC[idx];
	dout.fE = din.fE[idx];
	dout.fW = din.fW[idx];
	dout.fS = din.fS[idx];
	dout.fN = din.fN[idx];
	dout.fNE = din.fNE[idx];
	dout.fNW = din.fNW[idx];
	dout.fSE = din.fSE[idx];
	dout.fSW = din.fSW[idx];
}

//
// Get macroscopic density rho and velocity v given a distribution fi, and
// a the node class node_type.
//
__device__ void inline getMacro(Dist fi, int node_type, float &rho, float2 &v)
{
	rho = fi.fC + fi.fE + fi.fW + fi.fS + fi.fN + fi.fNE + fi.fNW + fi.fSE + fi.fSW;
	if (node_type == GEO_INFLOW) {
		v.x = 0.1f;
		v.y = 0.0f;
	} else {
		v.x = (fi.fE + fi.fSE + fi.fNE - fi.fW - fi.fSW - fi.fNW) / rho;
		v.y = (fi.fN + fi.fNW + fi.fNE - fi.fS - fi.fSW - fi.fSE) / rho;
	}
}

//
// A kernel to update the position of tracer particles.
//
// Each thread updates the position of a single particle using Euler's algorithm.
//
__global__ void LBMUpdateTracerParticles(DistP cd, int *map, float *x, float *y)
{
	float rho;
	float2 pv;

	int gi = threadIdx.x + blockDim.x * blockIdx.x;
	float cx = x[gi];
	float cy = y[gi];

	int ix = (int)(cx);
	int iy = (int)(cy);

	// Sanity checks.
	if (iy < 0)
		iy = 0;

	if (ix < 0)
		ix = 0;

	if (ix > LAT_W-1)
		ix = LAT_W-1;

	if (iy > LAT_H-1)
		iy = LAT_H-1;

	int dix = ix + LAT_W*iy;

	Dist fc;
	getDist(fc, cd, dix);
	getMacro(fc, map[dix], rho, pv);

	cx = cx + pv.x * DT;
	cy = cy + pv.y * DT;

	// Periodic boundary conditions.
	if (cx > LAT_W)
		cx = 0.0f;

	if (cy > LAT_H)
		cy = 0.0f;

	if (cx < 0.0f)
		cx = (float)LAT_W;

	if (cy < 0.0f)
		cy = (float)LAT_H;

	x[gi] = cx;
	y[gi] = cy;
}

//
// Performs the relaxation step in the BGK model given the density rho,
// the velocity v and the distribution fi.
//
__device__ void inline BGK_relaxate(float rho, float2 v, Dist &fi, int *map, int gi)
{
	// relaxation
	float Cusq = -1.5f * (v.x*v.x + v.y*v.y);
	Dist feq;

	feq.fC = rho * (1.0f + Cusq) * 4.0f/9.0f;
	feq.fN = rho * (1.0f + Cusq + 3.0f*v.y + 4.5f*v.y*v.y) / 9.0f;
	feq.fE = rho * (1.0f + Cusq + 3.0f*v.x + 4.5f*v.x*v.x) / 9.0f;
	feq.fS = rho * (1.0f + Cusq - 3.0f*v.y + 4.5f*v.y*v.y) / 9.0f;
	feq.fW = rho * (1.0f + Cusq - 3.0f*v.x + 4.5f*v.x*v.x) / 9.0f;
	feq.fNE = rho * (1.0f + Cusq + 3.0f*(v.x+v.y) + 4.5f*(v.x+v.y)*(v.x+v.y)) / 36.0f;
	feq.fSE = rho * (1.0f + Cusq + 3.0f*(v.x-v.y) + 4.5f*(v.x-v.y)*(v.x-v.y)) / 36.0f;
	feq.fSW = rho * (1.0f + Cusq + 3.0f*(-v.x-v.y) + 4.5f*(v.x+v.y)*(v.x+v.y)) / 36.0f;
	feq.fNW = rho * (1.0f + Cusq + 3.0f*(-v.x+v.y) + 4.5f*(-v.x+v.y)*(-v.x+v.y)) / 36.0f;

	if (map[gi] == GEO_FLUID) {
		fi.fC += (feq.fC - fi.fC) / tau;
		fi.fE += (feq.fE - fi.fE) / tau;
		fi.fW += (feq.fW - fi.fW) / tau;
		fi.fS += (feq.fS - fi.fS) / tau;
		fi.fN += (feq.fN - fi.fN) / tau;
		fi.fSE += (feq.fSE - fi.fSE) / tau;
		fi.fNE += (feq.fNE - fi.fNE) / tau;
		fi.fSW += (feq.fSW - fi.fSW) / tau;
		fi.fNW += (feq.fNW - fi.fNW) / tau;
	} else if (map[gi] == GEO_INFLOW) {
		fi.fC  = feq.fC;
		fi.fE  = feq.fE;
		fi.fW  = feq.fW;
		fi.fS  = feq.fS;
		fi.fN  = feq.fN;
		fi.fSE = feq.fSE;
		fi.fNE = feq.fNE;
		fi.fSW = feq.fSW;
		fi.fNW = feq.fNW;
	} else if (map[gi] == GEO_WALL) {
		float t;
		t = fi.fE;
		fi.fE = fi.fW;
		fi.fW = t;

		t = fi.fNW;
		fi.fNW = fi.fSE;
		fi.fSE = t;

		t = fi.fNE;
		fi.fNE = fi.fSW;
		fi.fSW = t;

		t = fi.fN;
		fi.fN = fi.fS;
		fi.fS = t;
	}
}

// TODO:
// - try having dummy nodes as the edges of the lattice to avoid divergent threads

__global__ void LBMCollideAndPropagate(int *map, DistP cd, DistP od, float *orho, float *ovx, float *ovy)
{
	int tix = threadIdx.x;
	int ti = tix + blockIdx.x * blockDim.x;
	int gi = ti + LAT_W*blockIdx.y;

	// shared variables for in-block propagation
	__shared__ float fo_E[BLOCK_SIZE];
	__shared__ float fo_W[BLOCK_SIZE];
	__shared__ float fo_SE[BLOCK_SIZE];
	__shared__ float fo_SW[BLOCK_SIZE];
	__shared__ float fo_NE[BLOCK_SIZE];
	__shared__ float fo_NW[BLOCK_SIZE];

	// cache the distribution in local variables
	Dist fi;
	getDist(fi, cd, gi);

	// macroscopic quantities for the current cell
	float rho;
	float2 v;
	getMacro(fi, map[gi], rho, v);

	// only save the macroscopic quantities if requested to do so
	if (orho != NULL) {
		orho[gi] = rho;
		ovx[gi] = v.x;
		ovy[gi] = v.y;
	}

	BGK_relaxate(rho, v, fi, map, gi);

	// update the 0-th direction distribution
	od.fC[gi] = fi.fC;

	// N + S propagation (global memory)
	if (blockIdx.y > 0)			od.fS[gi-LAT_W] = fi.fS;
	if (blockIdx.y < LAT_H-1)	od.fN[gi+LAT_W] = fi.fN;

	// E propagation in shared memory
	if (tix < blockDim.x-1) {
		fo_E[tix+1] = fi.fE;
		fo_NE[tix+1] = fi.fNE;
		fo_SE[tix+1] = fi.fSE;
	// E propagation in global memory (at block boundary)
	} else if (ti < LAT_W) {
		od.fE[gi+1] = fi.fE;
		if (blockIdx.y > 0)			od.fSE[gi-LAT_W+1] = fi.fSE;
		if (blockIdx.y < LAT_H-1)	od.fNE[gi+LAT_W+1] = fi.fNE;
	}

	// W propagation in shared memory
	if (tix > 0) {
		fo_W[tix-1] = fi.fW;
		fo_NW[tix-1] = fi.fNW;
		fo_SW[tix-1] = fi.fSW;
	// W propagation in global memory (at block boundary)
	} else if (ti > 0) {
		od.fW[gi-1] = fi.fW;
		if (blockIdx.y > 0)			od.fSW[gi-LAT_W-1] = fi.fSW;
		if (blockIdx.y < LAT_H-1)	od.fNW[gi+LAT_W-1] = fi.fNW;
	}

	__syncthreads();

	// the leftmost thread is not updated in this block
	if (tix > 0) {
		od.fE[gi] = fo_E[tix];
		if (blockIdx.y > 0)			od.fSE[gi-LAT_W] = fo_SE[tix];
		if (blockIdx.y < LAT_H-1)	od.fNE[gi+LAT_W] = fo_NE[tix];
	}

	// the rightmost thread is not updated in this block
	if (tix < blockDim.x-1) {
		od.fW[gi] = fo_W[tix];
		if (blockIdx.y > 0)			od.fSW[gi-LAT_W] = fo_SW[tix];
		if (blockIdx.y < LAT_H-1)	od.fNW[gi+LAT_W] = fo_NW[tix];
	}
}

