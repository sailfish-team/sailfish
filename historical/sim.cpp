#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include "sim.h"
#include "vis.h"

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
	struct SimState state;

	SimInit(&state);
	SDLInit();

	int iter = 0;

	SDL_Event event;
	bool quit = false;
	bool mouse = false;
	bool update_map = false;
	int last_x, last_y;

	struct timespec t1, t2;
	clock_gettime(CLOCK_REALTIME, &t1);

	while (!quit) {
		SimUpdate(iter, state);
		clock_gettime(CLOCK_REALTIME, &t2);

		if (iter % 100 == 0) {
			long timediff = (t2.tv_sec - t1.tv_sec) * 1000000000 + (t2.tv_nsec - t1.tv_nsec);

			visualize(&state, LAT_H * LAT_W * 100 * 1000000000.0f / timediff);
			clock_gettime(CLOCK_REALTIME, &t1);
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
			SimUpdateMap(state);
			update_map = false;
		}

		iter++;
	}

	SDL_Quit();
	SimCleanup(&state);
	return 0;
}
