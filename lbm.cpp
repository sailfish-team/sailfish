#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <hdf5.h>

#include "sim.h"

#define M_FLUID 0
#define M_WALL 1
#define M_SETU 2

const int size_i = LAT_W*LAT_H*sizeof(int);
const int size_f = LAT_W*LAT_H*sizeof(float);

const float visc = 0.01f;
float tau;

#define idx(x,y) ((y)*LAT_W+(x))

void allocate(struct SimState *state)
{
	int x, y, i;

	state->map = (int*)calloc(LAT_W*LAT_H, sizeof(int));
	state->vx = (float*)malloc(size_f);
	state->vy = (float*)malloc(size_f);
	state->rho = (float*)malloc(size_f);

	for (i = 0; i < 9; i++) {
		state->lat[i] = (float*)malloc(size_f);
	}

/*	ltc = (float***)calloc(mx, sizeof(float **));
	map = (char**)calloc(mx, sizeof(char*));

	for (x = 0; x < mx; x++) {
		ltc[x] = (float**)calloc(my, sizeof(float *));
		map[x] = (char*)calloc(my, sizeof(char));

		for (y = 0; y < my; y++) {
			ltc[x][y] = (float*)calloc(9, sizeof(float));
		}
	}*/
}

void init(struct SimState *state)
{
	int x, y, i;

	for (x = 0; x < LAT_W; x++) {
		for (y = 0; y < LAT_H; y++) {
			i = idx(x,y);
			state->lat[0][i] = 4.0f/9.0f;
			state->lat[1][i] =
			state->lat[2][i] =
			state->lat[3][i] =
			state->lat[4][i] = 1.0f/9.0f;
			state->lat[5][i] =
			state->lat[6][i] =
			state->lat[7][i] =
			state->lat[8][i] = 1.0f/36.0f;
		}
	}

	for (x = 0; x < LAT_W; x++) {
		state->map[idx(x,0)] = M_WALL;
	}

	for (y = 0; y < LAT_H; y++) {
		state->map[idx(0,y)] = state->map[idx(LAT_W-1,y)] = M_WALL;
	}

	for (x = 0; x < LAT_W; x++) {
		state->map[idx(x,LAT_H-1)] = M_SETU;
	}
}

void propagate(struct SimState state)
{
	int x, y;

	// west
	for (x = 0; x < LAT_W-1; x++) {
		for (y = 0; y < LAT_H; y++) {
			state.lat[4][idx(x,y)] = state.lat[4][idx(x+1,y)];
		}
	}

	// north-west
	for (x = 0; x < LAT_W-1; x++) {
		for (y = LAT_H-1; y > 0; y--) {
			state.lat[8][idx(x,y)] = state.lat[8][idx(x+1,y-1)];
		}
	}

	// north-east
	for (x = LAT_W-1; x > 0; x--) {
		for (y = LAT_H-1; y > 0; y--) {
			state.lat[5][idx(x,y)] = state.lat[5][idx(x-1,y-1)];
		}
	}

	// north
	for (x = 0; x < LAT_W; x++) {
		for (y = LAT_H-1; y > 0; y--) {
			state.lat[1][idx(x,y)] = state.lat[1][idx(x,y-1)];
		}
	}

	// south
	for (x = 0; x < LAT_W; x++) {
		for (y = 0; y < LAT_H-1; y++) {
			state.lat[3][idx(x,y)] = state.lat[3][idx(x,y+1)];
		}
	}

	// south-west
	for (x = 0; x < LAT_W-1; x++) {
		for (y = 0; y < LAT_H-1; y++) {
			state.lat[7][idx(x,y)] = state.lat[7][idx(x+1,y+1)];
		}
	}

	// south-east
	for (x = LAT_W-1; x > 0; x--) {
		for (y = 0; y < LAT_H-1; y++) {
			state.lat[6][idx(x,y)] = state.lat[6][idx(x-1,y+1)];
		}
	}

	// east
	for (x = LAT_W-1; x > 0; x--) {
		for (y = 0; y < LAT_H; y++) {
			state.lat[2][idx(x,y)] = state.lat[2][idx(x-1,y)];
		}
	}
}

void get_macro(struct SimState state, int x, int y, float &rho, float &vx, float &vy)
{
	int i;
	rho = 0.0;

	int gi = idx(x,y);

	for (i = 0; i < 9; i++) {
		rho += state.lat[i][gi];
	}

	if (state.map[gi] == M_FLUID || state.map[gi] == M_WALL) {
		vx = (state.lat[2][gi] + state.lat[5][gi] + state.lat[6][gi] - state.lat[8][gi] - state.lat[4][gi] - state.lat[7][gi])/rho;
		vy = (state.lat[1][gi] + state.lat[5][gi] + state.lat[8][gi] - state.lat[7][gi] - state.lat[3][gi] - state.lat[6][gi])/rho;
	} else {
		vx = 0.1f;
		vy = 0.0f;
	}
}

//
// Directions are:
//
//    8  1  5
// ^  4  0  2
// |  7  3  6
//  ->

void relaxate(struct SimState state)
{
	int x, y, i, gi;

	for (x = 0; x < LAT_W; x++) {
		for (y = 0; y < LAT_H; y++) {

			gi = idx(x, y);

			if (state.map[idx(x,y)] != M_WALL) {
				float vx, vy, rho;
				get_macro(state, x, y, rho, vx, vy);

				state.vx[gi] = vx;
				state.vy[gi] = vy;
				state.rho[gi] = rho;

				float Cusq = -1.5f * (vx*vx + vy*vy);
				float feq[9];

				feq[0] = rho * (1.0f + Cusq) * 4.0f/9.0f;
				feq[1] = rho * (1.0f + Cusq + 3.0f*vy + 4.5f*vy*vy) / 9.0f;
				feq[2] = rho * (1.0f + Cusq + 3.0f*vx + 4.5f*vx*vx) / 9.0f;
				feq[3] = rho * (1.0f + Cusq - 3.0f*vy + 4.5f*vy*vy) / 9.0f;
				feq[4] = rho * (1.0f + Cusq - 3.0f*vx + 4.5f*vx*vx) / 9.0f;
				feq[5] = rho * (1.0f + Cusq + 3.0f*(vx+vy) + 4.5f*(vx+vy)*(vx+vy)) / 36.0f;
				feq[6] = rho * (1.0f + Cusq + 3.0f*(vx-vy) + 4.5f*(vx-vy)*(vx-vy)) / 36.0f;
				feq[7] = rho * (1.0f + Cusq + 3.0f*(-vx-vy) + 4.5f*(vx+vy)*(vx+vy)) / 36.0f;
				feq[8] = rho * (1.0f + Cusq + 3.0f*(-vx+vy) + 4.5f*(-vx+vy)*(-vx+vy)) / 36.0f;

				if (state.map[idx(x,y)] == M_FLUID) {
					for (i = 0; i < 9; i++) {
						state.lat[i][idx(x,y)] += (feq[i] - state.lat[i][idx(x,y)]) / tau;
					}
				} else {
					for (i = 0; i < 9; i++) {
						state.lat[i][idx(x,y)] = feq[i];
					}
				}
			} else {
				float tmp;
				tmp = state.lat[2][idx(x,y)];
				state.lat[2][idx(x,y)] = state.lat[4][idx(x,y)];
				state.lat[4][idx(x,y)] = tmp;

				tmp = state.lat[1][idx(x,y)];
				state.lat[1][idx(x,y)] = state.lat[3][idx(x,y)];
				state.lat[3][idx(x,y)] = tmp;

				tmp = state.lat[8][idx(x,y)];
				state.lat[8][idx(x,y)] = state.lat[6][idx(x,y)];
				state.lat[6][idx(x,y)] = tmp;

				tmp = state.lat[7][idx(x,y)];
				state.lat[7][idx(x,y)] = state.lat[5][idx(x,y)];
				state.lat[6][idx(x,y)] = tmp;
			}
		}
	}
}

void SimInit(struct SimState *state)
{
	tau = (6.0f*visc + 1.0f)/2.0f;

	int Re;
	Re=(int)((LAT_W-1)*0.1f/((2.0f*tau-1.0f)/6.0f)+0.5f);

	printf("visc = %f\n", visc);
	printf("tau = %f\n", tau);
	printf("Re = %d\n", Re);

	allocate(state);
	init(state);
}

void SimCleanup(struct SimState *state)
{
	return;
}

void SimUpdate(int iter, struct SimState state)
{
	relaxate(state);
	propagate(state);
}

void SimUpdateMap(struct SimState state)
{
	return;
}

