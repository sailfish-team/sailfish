#ifndef __SIM_H_
#define __SIM_H_ 1

#define GEO_FLUID 0
#define GEO_WALL 1
#define GEO_INFLOW 2

#define LAT_H 128
#define LAT_W 128
#define BLOCK_SIZE 64

struct Dist {
	float *fC, *fE, *fW, *fS, *fN, *fSE, *fSW, *fNE, *fNW;
};

struct SimState {
	int *map, *dmap;

	// macroscopic quantities on the video card
	float *dvx, *dvy, *drho;

	// macroscopic quantities in RAM
	float *vx, *vy, *rho;

	float *lat[9];
	Dist d1, d2;
};

void SimInit(struct SimState *state);
void SimCleanup(struct SimState *state);
void SimUpdate(int iter, struct SimState state);
void SimUpdateMap(struct SimState state);

#endif  /* __SIM_H_ */
