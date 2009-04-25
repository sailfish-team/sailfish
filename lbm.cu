#include <cstdio>
#include <cstdlib>
#include <cmath>

#define GEO_FLUID 0
#define GEO_WALL 1
#define GEO_INFLOW 2

#define LAT_H 128
#define LAT_W 128

#define BLOCK_SIZE 64

const float visc = 0.01f;				// viscosity
__constant__ float tau;					// relaxation time
__constant__ int latH = LAT_H;			// lattice height

struct Dist {
	float *fC, *fE, *fW, *fS, *fN, *fSE, *fSW, *fNE, *fNW;
};

// TODO:
// - try having dummy nodes as the edges of the lattice to avoid divergent threads

__global__ void LBMCollideAndPropagate(int *map, Dist cd, Dist od, float *orho, float *ovx, float *ovy)
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

void output(int snum, float *vx, float *vy, float *rho)
{
	int x, y, i;
	char name[128];
	FILE *fp;

	sprintf(name, "out%05d.dat", snum);
	fp = fopen(name, "w");

	i = 0;
	sprintf(name, "t%d", snum);
	for (y = 0; y < LAT_H; y++) {
		for (x = 0; x < LAT_W; x++) {
			fprintf(fp, "%d %d %f %f %f\n", x, y, rho[i], vx[i], vy[i]);
			i++;
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

int main(int argc, char **argv)
{
	int i;

	// setup relaxation time
	float tmp = (6.0f*visc + 1.0f)/2.0f;
	cudaMemcpyToSymbol(tau, &tmp, sizeof(float));

	int size_i = LAT_W*LAT_H*sizeof(int);
	int size_f = LAT_W*LAT_H*sizeof(float);

	int *map, *dmap;
	map = (int*)calloc(LAT_W*LAT_H, sizeof(int));
	cudaMalloc((void**)&dmap, size_i);

	// macroscopic quantities on the video card
	float *dvx, *dvy, *drho;
	cudaMalloc((void**)&dvx, size_f);
	cudaMalloc((void**)&dvy, size_f);
	cudaMalloc((void**)&drho, size_f);

	// macroscopic quantities in RAM
	float *vx, *vy, *rho;
	vx = (float*)malloc(size_f);
	vy = (float*)malloc(size_f);
	rho = (float*)malloc(size_f);

	float *lat[9];
	Dist d1, d2;

	cudaMalloc((void**)&d1.fC, size_f);
	cudaMalloc((void**)&d1.fE, size_f);
	cudaMalloc((void**)&d1.fW, size_f);
	cudaMalloc((void**)&d1.fN, size_f);
	cudaMalloc((void**)&d1.fS, size_f);
	cudaMalloc((void**)&d1.fNE, size_f);
	cudaMalloc((void**)&d1.fSE, size_f);
	cudaMalloc((void**)&d1.fNW, size_f);
	cudaMalloc((void**)&d1.fSW, size_f);

	cudaMalloc((void**)&d2.fC, size_f);
	cudaMalloc((void**)&d2.fE, size_f);
	cudaMalloc((void**)&d2.fW, size_f);
	cudaMalloc((void**)&d2.fN, size_f);
	cudaMalloc((void**)&d2.fS, size_f);
	cudaMalloc((void**)&d2.fNE, size_f);
	cudaMalloc((void**)&d2.fSE, size_f);
	cudaMalloc((void**)&d2.fNW, size_f);
	cudaMalloc((void**)&d2.fSW, size_f);

	for (i = 0; i < 9; i++) {
		lat[i] = (float*)malloc(size_f);
	}

	for (i = 0; i < LAT_W*LAT_H; i++) {
		lat[0][i] = 4.0/9.0;
		lat[1][i] = lat[2][i] = lat[3][i] = lat[4][i] = 1.0/9.0;
		lat[5][i] = lat[6][i] = lat[7][i] = lat[8][i] = 1.0/36.0;
	}

	for (i = 0; i < LAT_W; i++) {
		map[i] = GEO_WALL;
	}

	for (i = 0; i < LAT_H; i++) {
		map[i*LAT_H] = map[LAT_W-1 + i*LAT_H] = GEO_WALL;
	}

	for (i = 0; i < LAT_W; i++) {
		map[(LAT_H-1)*LAT_W + i] = GEO_INFLOW;
	}

	cudaMemcpy(dmap, map, size_i, cudaMemcpyHostToDevice);
	cudaMemcpy(d1.fC, lat[0], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(d1.fN, lat[1], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(d1.fS, lat[2], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(d1.fE, lat[3], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(d1.fW, lat[4], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(d1.fNE, lat[5], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(d1.fNW, lat[6], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(d1.fSE, lat[7], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(d1.fSW, lat[8], size_f, cudaMemcpyHostToDevice);

	cudaMemcpy(d2.fC, lat[0], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(d2.fN, lat[1], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(d2.fS, lat[2], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(d2.fE, lat[3], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(d2.fW, lat[4], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(d2.fNE, lat[5], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(d2.fNW, lat[6], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(d2.fSE, lat[7], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(d2.fSW, lat[8], size_f, cudaMemcpyHostToDevice);
	dim3 grid;

	grid.x = LAT_W/BLOCK_SIZE;
	grid.y = LAT_H;

	for (int iter = 0; iter < 10000; iter++) {

		if (iter % 100 == 0) {
			LBMCollideAndPropagate<<<grid, BLOCK_SIZE, BLOCK_SIZE*6*sizeof(float)>>>(dmap, d1, d2, drho, dvx, dvy);
			cudaMemcpy(vx, dvx, size_f, cudaMemcpyDeviceToHost);
			cudaMemcpy(vy, dvy, size_f, cudaMemcpyDeviceToHost);
			cudaMemcpy(rho, drho, size_f, cudaMemcpyDeviceToHost);
			output(iter, vx, vy, rho);
		} else {
			// A-B access pattern with swapped distributions d1, d2
			if (iter % 2 == 0) {
				LBMCollideAndPropagate<<<grid, BLOCK_SIZE, BLOCK_SIZE*6*sizeof(float)>>>(dmap, d1, d2, drho, dvx, dvy);
			} else {
				LBMCollideAndPropagate<<<grid, BLOCK_SIZE, BLOCK_SIZE*6*sizeof(float)>>>(dmap, d2, d1, drho, dvx, dvy);
			}
		}
	}

	free(map);
	cudaFree(dmap);
	for (i = 0; i < 0; i++) {
		free(lat[i]);
	}

	cudaFree(dvx);
	cudaFree(dvy);
	cudaFree(drho);

	cudaFree(d1.fC);
	cudaFree(d1.fE);
	cudaFree(d1.fW);
	cudaFree(d1.fS);
	cudaFree(d1.fN);
	cudaFree(d1.fNE);
	cudaFree(d1.fNW);
	cudaFree(d1.fSE);
	cudaFree(d1.fSW);

	cudaFree(d2.fC);
	cudaFree(d2.fE);
	cudaFree(d2.fW);
	cudaFree(d2.fS);
	cudaFree(d2.fN);
	cudaFree(d2.fNE);
	cudaFree(d2.fNW);
	cudaFree(d2.fSE);
	cudaFree(d2.fSW);

	return 0;
}
