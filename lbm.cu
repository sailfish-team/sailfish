#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "sim.h"
#include "vis.h"

const int size_i = LAT_W*LAT_H*sizeof(int);
const int size_f = LAT_W*LAT_H*sizeof(float);

const float visc = 0.01f;				// viscosity
__constant__ float tau;					// relaxation time
__constant__ int latH = LAT_H;			// lattice height

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

void SimInit(struct SimState *state)
{
	int i;

	// setup relaxation time
	float tmp = (6.0f*visc + 1.0f)/2.0f;
	cudaMemcpyToSymbol(tau, &tmp, sizeof(float));

	state->map = (int*)calloc(LAT_W*LAT_H, sizeof(int));
	cudaMalloc((void**)&state->dmap, size_i);

	cudaMalloc((void**)&state->dvx, size_f);
	cudaMalloc((void**)&state->dvy, size_f);
	cudaMalloc((void**)&state->drho, size_f);

	state->vx = (float*)malloc(size_f);
	state->vy = (float*)malloc(size_f);
	state->rho = (float*)malloc(size_f);

	cudaMalloc((void**)&state->d1.fC, size_f);
	cudaMalloc((void**)&state->d1.fE, size_f);
	cudaMalloc((void**)&state->d1.fW, size_f);
	cudaMalloc((void**)&state->d1.fN, size_f);
	cudaMalloc((void**)&state->d1.fS, size_f);
	cudaMalloc((void**)&state->d1.fNE, size_f);
	cudaMalloc((void**)&state->d1.fSE, size_f);
	cudaMalloc((void**)&state->d1.fNW, size_f);
	cudaMalloc((void**)&state->d1.fSW, size_f);

	cudaMalloc((void**)&state->d2.fC, size_f);
	cudaMalloc((void**)&state->d2.fE, size_f);
	cudaMalloc((void**)&state->d2.fW, size_f);
	cudaMalloc((void**)&state->d2.fN, size_f);
	cudaMalloc((void**)&state->d2.fS, size_f);
	cudaMalloc((void**)&state->d2.fNE, size_f);
	cudaMalloc((void**)&state->d2.fSE, size_f);
	cudaMalloc((void**)&state->d2.fNW, size_f);
	cudaMalloc((void**)&state->d2.fSW, size_f);

	for (i = 0; i < 9; i++) {
		state->lat[i] = (float*)malloc(size_f);
	}

	for (i = 0; i < LAT_W*LAT_H; i++) {
		state->lat[0][i] = 4.0/9.0;
		state->lat[1][i] = state->lat[2][i] = state->lat[3][i] = state->lat[4][i] = 1.0/9.0;
		state->lat[5][i] = state->lat[6][i] = state->lat[7][i] = state->lat[8][i] = 1.0/36.0;
	}

	// bottom
	for (i = 0; i < LAT_W; i++) {
		state->map[i] = GEO_WALL;
	}

	// left / right
	for (i = 0; i < LAT_H; i++) {
		state->map[i*LAT_W] = state->map[LAT_W-1 + i*LAT_W] = GEO_WALL;
	}

	// top
	for (i = 0; i < LAT_W; i++) {
		state->map[(LAT_H-1)*LAT_W + i] = GEO_INFLOW;
	}

	cudaMemcpy(state->dmap, state->map, size_i, cudaMemcpyHostToDevice);
	cudaMemcpy(state->d1.fC, state->lat[0], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(state->d1.fN, state->lat[1], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(state->d1.fS, state->lat[2], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(state->d1.fE, state->lat[3], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(state->d1.fW, state->lat[4], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(state->d1.fNE, state->lat[5], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(state->d1.fNW, state->lat[6], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(state->d1.fSE, state->lat[7], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(state->d1.fSW, state->lat[8], size_f, cudaMemcpyHostToDevice);

	cudaMemcpy(state->d2.fC, state->lat[0], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(state->d2.fN, state->lat[1], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(state->d2.fS, state->lat[2], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(state->d2.fE, state->lat[3], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(state->d2.fW, state->lat[4], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(state->d2.fNE, state->lat[5], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(state->d2.fNW, state->lat[6], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(state->d2.fSE, state->lat[7], size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(state->d2.fSW, state->lat[8], size_f, cudaMemcpyHostToDevice);
}

void SimCleanup(struct SimState *state)
{
	int i;

	free(state->map);
	cudaFree(state->dmap);
	for (i = 0; i < 8; i++) {
		free(state->lat[i]);
	}

	cudaFree(state->dvx);
	cudaFree(state->dvy);
	cudaFree(state->drho);

	cudaFree(state->d1.fC);
	cudaFree(state->d1.fE);
	cudaFree(state->d1.fW);
	cudaFree(state->d1.fS);
	cudaFree(state->d1.fN);
	cudaFree(state->d1.fNE);
	cudaFree(state->d1.fNW);
	cudaFree(state->d1.fSE);
	cudaFree(state->d1.fSW);

	cudaFree(state->d2.fC);
	cudaFree(state->d2.fE);
	cudaFree(state->d2.fW);
	cudaFree(state->d2.fS);
	cudaFree(state->d2.fN);
	cudaFree(state->d2.fNE);
	cudaFree(state->d2.fNW);
	cudaFree(state->d2.fSE);
	cudaFree(state->d2.fSW);
}

int main(int argc, char **argv)
{
	dim3 grid;
	grid.x = LAT_W/BLOCK_SIZE;
	grid.y = LAT_H;

	struct SimState state;

	SimInit(&state);
	SDLInit();

	int iter = 0;

	SDL_Event event;
	bool quit = false;
	bool mouse = false;
	bool update_map = false;
	int last_x, last_y;

	while (!quit) {

		if (iter % 100 == 0) {
			LBMCollideAndPropagate<<<grid, BLOCK_SIZE, BLOCK_SIZE*6*sizeof(float)>>>(state.dmap, state.d1, state.d2, state.drho, state.dvx, state.dvy);
			cudaMemcpy(state.vx, state.dvx, size_f, cudaMemcpyDeviceToHost);
			cudaMemcpy(state.vy, state.dvy, size_f, cudaMemcpyDeviceToHost);
			cudaMemcpy(state.rho, state.drho, size_f, cudaMemcpyDeviceToHost);
//			output(iter, state.vx, state.vy, state.rho);
			visualize(&state);
		} else {
			// A-B access pattern with swapped distributions d1, d2
			if (iter % 2 == 0) {
				LBMCollideAndPropagate<<<grid, BLOCK_SIZE, BLOCK_SIZE*6*sizeof(float)>>>(state.dmap, state.d1, state.d2, state.drho, state.dvx, state.dvy);
			} else {
				LBMCollideAndPropagate<<<grid, BLOCK_SIZE, BLOCK_SIZE*6*sizeof(float)>>>(state.dmap, state.d2, state.d1, state.drho, state.dvx, state.dvy);
			}
		}

	    while (SDL_PollEvent(&event)) {
			switch (event.type) {
			case SDL_QUIT:
				quit = true;
				break;
			case SDL_KEYDOWN:
				quit = true;
				break;
			case SDL_MOUSEBUTTONUP:
				if (event.button.button == SDL_BUTTON_LEFT) {
					mouse = false;
					last_x = event.button.x / VIS_BLOCK_SIZE;
					last_y = event.button.y / VIS_BLOCK_SIZE;
					state.map[(LAT_H-last_y-1) * LAT_W + last_x] = GEO_WALL;
					update_map = true;
				}
				break;

			case SDL_MOUSEBUTTONDOWN:
				if (event.button.button == SDL_BUTTON_LEFT) {
					mouse = true;
					last_x = event.button.x / VIS_BLOCK_SIZE;
					last_y = event.button.y / VIS_BLOCK_SIZE;
					state.map[(LAT_H-last_y-1) * LAT_W + last_x] = GEO_WALL;
					update_map = true;
				}
				break;

			case SDL_MOUSEMOTION:
				if (mouse) {
					last_x = event.motion.x / VIS_BLOCK_SIZE;
					last_y = event.motion.y / VIS_BLOCK_SIZE;
					state.map[(LAT_H-last_y-1) * LAT_W + last_x] = GEO_WALL;
					update_map = true;
				}
			}
		}

		if (update_map) {
			cudaMemcpy(state.dmap, state.map, size_i, cudaMemcpyHostToDevice);
			update_map = false;
		}

		iter++;
	}

	SDL_Quit();
	SimCleanup(&state);
	return 0;
}
