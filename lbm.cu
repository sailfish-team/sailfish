// the following additional constants need to be defined:
// LAT_H, LAT_W, BLOCK_SIZE, GEO_FLUID, GEO_WALL, GEO_INFLOW

__constant__ float tau;			// relaxation time

struct DistP {
	float *fC, *fE, *fW, *fS, *fN, *fSE, *fSW, *fNE, *fNW;
};

struct Dist {
	float fC, fE, fW, fS, fN, fSE, fSW, fNE, fNW;
};

// TODO:
// - try having dummy nodes as the edges of the lattice to avoid divergent threads

__global__ void LBMCollideAndPropagate(int *map, DistP cd, DistP od, float *orho, float *ovx, float *ovy)
{
	int tix = threadIdx.x;
	int ti = tix + blockIdx.x * blockDim.x;
	int gi = ti + LAT_W*blockIdx.y;

	// equilibrium distributions
	Dist feq, fi;
	float rho;
	float2 v;

	// shared variables for in-block propagation
	__shared__ float fo_E[BLOCK_SIZE];
	__shared__ float fo_W[BLOCK_SIZE];
	__shared__ float fo_SE[BLOCK_SIZE];
	__shared__ float fo_SW[BLOCK_SIZE];
	__shared__ float fo_NE[BLOCK_SIZE];
	__shared__ float fo_NW[BLOCK_SIZE];

	// cache the distribution in local variables
	fi.fC = cd.fC[gi];
	fi.fE = cd.fE[gi];
	fi.fW = cd.fW[gi];
	fi.fS = cd.fS[gi];
	fi.fN = cd.fN[gi];
	fi.fNE = cd.fNE[gi];
	fi.fNW = cd.fNW[gi];
	fi.fSE = cd.fSE[gi];
	fi.fSW = cd.fSW[gi];

	// macroscopic quantities for the current cell
	rho = fi.fC + fi.fE + fi.fW + fi.fS + fi.fN + fi.fNE + fi.fNW + fi.fSE + fi.fSW;
	if (map[gi] == GEO_INFLOW) {
		v.x = 0.1f;
		v.y = 0.0f;
	} else {
		v.x = (fi.fE + fi.fSE + fi.fNE - fi.fW - fi.fSW - fi.fNW) / rho;
		v.y = (fi.fN + fi.fNW + fi.fNE - fi.fS - fi.fSW - fi.fSE) / rho;
	}

	if (orho != NULL) {
		orho[gi] = rho;
		ovx[gi] = v.x;
		ovy[gi] = v.y;
	}

	// relaxation
	float Cusq = -1.5f * (v.x*v.x + v.y*v.y);

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

