// the following constants need to be dynamically
// determined: LAT_H, LAT_W, BLOCK_SIZE

__constant__ float tau;			// relaxation time

#define GEO_FLUID 0
#define GEO_WALL 1
#define GEO_INFLOW 2

struct Dist {
	float *fC, *fE, *fW, *fS, *fN, *fSE, *fSW, *fNE, *fNW;
};

// TODO:
// - try having dummy nodes as the edges of the lattice to avoid divergent threads

__global__ void LBMCollideAndPropagate(int *map, Dist cd, Dist od, float *orho, float *ovx, float *ovy)

//__global__ void LBMCollideAndPropagate(int *map,
//float *cd___fC, float *cd___fE, float *cd___fW, float *cd___fS, float *cd___fN, float *cd___fSE, float *cd___fSW,  float*cd___fNE,  float*cd___fNW,
//float *od___fC, float *od___fE, float *od___fW, float *od___fS, float *od___fN, float *od___fSE, float *od___fSW, float *od___fNE, float *od___fNW,
//float *orho, float *ovx, float *ovy)
{
	int tix = threadIdx.x;
	int ti = tix + blockIdx.x * blockDim.x;
	int gi = ti + LAT_W*blockIdx.y;

	// equilibrium distributions
	float feq_C, feq_N, feq_S, feq_E, feq_W, feq_NE, feq_NW, feq_SE, feq_SW;
	float fi_N, fi_S, fi_E, fi_W, fi_C, fi_NE, fi_NW, fi_SE, fi_SW;
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
	fi_C = cd.fC[gi];
	fi_E = cd.fE[gi];
	fi_W = cd.fW[gi];
	fi_S = cd.fS[gi];
	fi_N = cd.fN[gi];
	fi_NE = cd.fNE[gi];
	fi_NW = cd.fNW[gi];
	fi_SE = cd.fSE[gi];
	fi_SW = cd.fSW[gi];

	// macroscopic quantities for the current cell
	rho = fi_C + fi_E + fi_W + fi_S + fi_N + fi_NE + fi_NW + fi_SE + fi_SW;
	if (map[gi] == GEO_INFLOW) {
		v.x = 0.1f;
		v.y = 0.0f;
	} else {
		v.x = (fi_E + fi_SE + fi_NE - fi_W - fi_SW - fi_NW) / rho;
		v.y = (fi_N + fi_NW + fi_NE - fi_S - fi_SW - fi_SE) / rho;
	}

	if (orho != NULL) {
		orho[gi] = rho;
		ovx[gi] = v.x;
		ovy[gi] = v.y;
	}

	// relaxation
	float Cusq = -1.5f * (v.x*v.x + v.y*v.y);

	feq_C = rho * (1.0f + Cusq) * 4.0f/9.0f;
	feq_N = rho * (1.0f + Cusq + 3.0f*v.y + 4.5f*v.y*v.y) / 9.0f;
	feq_E = rho * (1.0f + Cusq + 3.0f*v.x + 4.5f*v.x*v.x) / 9.0f;
	feq_S = rho * (1.0f + Cusq - 3.0f*v.y + 4.5f*v.y*v.y) / 9.0f;
	feq_W = rho * (1.0f + Cusq - 3.0f*v.x + 4.5f*v.x*v.x) / 9.0f;
	feq_NE = rho * (1.0f + Cusq + 3.0f*(v.x+v.y) + 4.5f*(v.x+v.y)*(v.x+v.y)) / 36.0f;
	feq_SE = rho * (1.0f + Cusq + 3.0f*(v.x-v.y) + 4.5f*(v.x-v.y)*(v.x-v.y)) / 36.0f;
	feq_SW = rho * (1.0f + Cusq + 3.0f*(-v.x-v.y) + 4.5f*(v.x+v.y)*(v.x+v.y)) / 36.0f;
	feq_NW = rho * (1.0f + Cusq + 3.0f*(-v.x+v.y) + 4.5f*(-v.x+v.y)*(-v.x+v.y)) / 36.0f;

	if (map[gi] == GEO_FLUID) {
		fi_C += (feq_C - fi_C) / tau;
		fi_E += (feq_E - fi_E) / tau;
		fi_W += (feq_W - fi_W) / tau;
		fi_S += (feq_S - fi_S) / tau;
		fi_N += (feq_N - fi_N) / tau;
		fi_SE += (feq_SE - fi_SE) / tau;
		fi_NE += (feq_NE - fi_NE) / tau;
		fi_SW += (feq_SW - fi_SW) / tau;
		fi_NW += (feq_NW - fi_NW) / tau;
	} else if (map[gi] == GEO_INFLOW) {
		fi_C  = feq_C;
		fi_E  = feq_E;
		fi_W  = feq_W;
		fi_S  = feq_S;
		fi_N  = feq_N;
		fi_SE = feq_SE;
		fi_NE = feq_NE;
		fi_SW = feq_SW;
		fi_NW = feq_NW;
	} else if (map[gi] == GEO_WALL) {
		float t;
		t = fi_E;
		fi_E = fi_W;
		fi_W = t;

		t = fi_NW;
		fi_NW = fi_SE;
		fi_SE = t;

		t = fi_NE;
		fi_NE = fi_SW;
		fi_SW = t;

		t = fi_N;
		fi_N = fi_S;
		fi_S = t;
	}

	od.fC[gi] = fi_C;

	// N + S propagation (global memory)
	if (blockIdx.y > 0)			od.fS[gi-LAT_W] = fi_S;
	if (blockIdx.y < LAT_H-1)	od.fN[gi+LAT_W] = fi_N;

	// E propagation in shared memory
	if (tix < blockDim.x-1) {
		fo_E[tix+1] = fi_E;
		fo_NE[tix+1] = fi_NE;
		fo_SE[tix+1] = fi_SE;
	// E propagation in global memory (at block boundary)
	} else if (ti < LAT_W) {
		od.fE[gi+1] = fi_E;
		if (blockIdx.y > 0)			od.fSE[gi-LAT_W+1] = fi_SE;
		if (blockIdx.y < LAT_H-1)	od.fNE[gi+LAT_W+1] = fi_NE;
	}

	// W propagation in shared memory
	if (tix > 0) {
		fo_W[tix-1] = fi_W;
		fo_NW[tix-1] = fi_NW;
		fo_SW[tix-1] = fi_SW;
	// W propagation in global memory (at block boundary)
	} else if (ti > 0) {
		od.fW[gi-1] = fi_W;
		if (blockIdx.y > 0)			od.fSW[gi-LAT_W-1] = fi_SW;
		if (blockIdx.y < LAT_H-1)	od.fNW[gi+LAT_W-1] = fi_NW;
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

