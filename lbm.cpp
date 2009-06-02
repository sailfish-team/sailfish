#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <hdf5.h>

#define M_FLUID 0
#define M_WALL 1
#define M_SETU 2

const int mx = 128;
const int my = 128;

const float visc = 0.01f;
float tau;

float ***outbuf;
float ***ltc;
char **map;

void allocate()
{
	int x, y;

	ltc = (float***)calloc(mx, sizeof(float **));
	outbuf = (float***)calloc(mx, sizeof(float **));
	map = (char**)calloc(mx, sizeof(char*));

	for (x = 0; x < mx; x++) {
		ltc[x] = (float**)calloc(my, sizeof(float *));
		map[x] = (char*)calloc(my, sizeof(char));
		outbuf[x] = (float**)calloc(my, sizeof(float *));

		for (y = 0; y < my; y++) {
			ltc[x][y] = (float*)calloc(9, sizeof(float));
		}
	}

	outbuf[0][0] = (float*)calloc(mx*my*3, sizeof(float));

	for (x = 0; x < mx; x++) {
		for (y = 0; y < my; y++) {
			outbuf[x][y] = outbuf[0][0] + (x*my + y)*3;
		}
	}
}

void init()
{
	int x, y;

	for (x = 0; x < mx; x++) {
		for (y = 0; y < my; y++) {
			ltc[x][y][0] = 4.0f/9.0f;
			ltc[x][y][1] =
			ltc[x][y][2] =
			ltc[x][y][3] =
			ltc[x][y][4] = 1.0f/9.0f;
			ltc[x][y][5] =
			ltc[x][y][6] =
			ltc[x][y][7] =
			ltc[x][y][8] = 1.0f/36.0f;
		}
	}

	for (x = 0; x < mx; x++) {
		map[x][0] = M_WALL;
	}

	for (y = 0; y < my; y++) {
		map[0][y] = map[mx-1][y] = M_WALL;
	}

	for (x = 0; x < mx; x++) {
		map[x][my-1] = M_SETU;
	}
}

void propagate()
{
	int x, y;

	// west
	for (x = 0; x < mx-1; x++) {
		for (y = 0; y < my; y++) {
			ltc[x][y][4] = ltc[x+1][y][4];
		}
	}

	// north-west
	for (x = 0; x < mx-1; x++) {
		for (y = my-1; y > 0; y--) {
			ltc[x][y][8] = ltc[x+1][y-1][8];
		}
	}

	// north-east
	for (x = mx-1; x > 0; x--) {
		for (y = my-1; y > 0; y--) {
			ltc[x][y][5] = ltc[x-1][y-1][5];
		}
	}

	// north
	for (x = 0; x < mx; x++) {
		for (y = my-1; y > 0; y--) {
			ltc[x][y][1] = ltc[x][y-1][1];
		}
	}

	// south
	for (x = 0; x < mx; x++) {
		for (y = 0; y < my-1; y++) {
			ltc[x][y][3] = ltc[x][y+1][3];
		}
	}

	// south-west
	for (x = 0; x < mx-1; x++) {
		for (y = 0; y < my-1; y++) {
			ltc[x][y][7] = ltc[x+1][y+1][7];
		}
	}

	// south-east
	for (x = mx-1; x > 0; x--) {
		for (y = 0; y < my-1; y++) {
			ltc[x][y][6] = ltc[x-1][y+1][6];
		}
	}

	// east
	for (x = mx-1; x > 0; x--) {
		for (y = 0; y < my; y++) {
			ltc[x][y][2] = ltc[x-1][y][2];
		}
	}
}

void get_macro(int x, int y, float &rho, float &vx, float &vy)
{
	int i;
	rho = 0.0;

	for (i = 0; i < 9; i++) {
		rho += ltc[x][y][i];
	}

	if (map[x][y] == M_FLUID || map[x][y] == M_WALL) {
		vx = (ltc[x][y][2] + ltc[x][y][5] + ltc[x][y][6] - ltc[x][y][8] - ltc[x][y][4] - ltc[x][y][7])/rho;
		vy = (ltc[x][y][1] + ltc[x][y][5] + ltc[x][y][8] - ltc[x][y][7] - ltc[x][y][3] - ltc[x][y][6])/rho;
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

void relaxate()
{
	int x, y, i;

	for (x = 0; x < mx; x++) {
		for (y = 0; y < my; y++) {

			if (map[x][y] != M_WALL) {
				float vx, vy, rho;
				get_macro(x, y, rho, vx, vy);

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

				if (map[x][y] == M_FLUID) {
					for (i = 0; i < 9; i++) {
						ltc[x][y][i] += (feq[i] - ltc[x][y][i]) / tau;
					}
				} else {
					for (i = 0; i < 9; i++) {
						ltc[x][y][i] = feq[i];
					}
				}
			} else {
				float tmp;
				tmp = ltc[x][y][2];
				ltc[x][y][2] = ltc[x][y][4];
				ltc[x][y][4] = tmp;

				tmp = ltc[x][y][1];
				ltc[x][y][1] = ltc[x][y][3];
				ltc[x][y][3] = tmp;

				tmp = ltc[x][y][8];
				ltc[x][y][8] = ltc[x][y][6];
				ltc[x][y][6] = tmp;

				tmp = ltc[x][y][7];
				ltc[x][y][7] = ltc[x][y][5];
				ltc[x][y][5] = tmp;
			}
		}
	}
}

void output(int snum)
	//, hid_t file, hid_t dataspace, hid_t datatype)
{
	int x, y;
	char name[128];
	FILE *fp;

	sprintf(name, "out%05d.dat", snum);
	fp = fopen(name, "w");

	sprintf(name, "t%d", snum);
	for (x = 0; x < mx; x++) {
		for (y = 0; y < my; y++) {
			float vx, vy, rho;
			get_macro(x, y, rho, vx, vy);

/*			outbuf[x][y][0] = rho;
			outbuf[x][y][1] = vx;
			outbuf[x][y][2] = vy;
*/
//			printf("%x\n", &outbuf[x][y][0]);

//			fprintf(fp, "%d %d %f %f %f | %f %f %f %f %f %f %f %f %f\n", x, y, rho, vx, vy,
//					ltc[x][y][0], ltc[x][y][2], ltc[x][y][3], ltc[x][y][4], ltc[x][y][1],
//					ltc[x][y][6], ltc[x][y][7], ltc[x][y][8], ltc[x][y][5]);
			fprintf(fp, "%d %d %f %f %f\n", x, y, rho, vx, vy);
		}
		fprintf(fp, "\n");
	}
//	hid_t dataset = H5Dcreate(file, name, datatype, dataspace, H5P_DEFAULT);
//    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &outbuf[0][0][0]);
//	H5Dclose(dataset);

	fclose(fp);
}


int main(int argc, char **argv)
{
	int st = 0;
	int Re;

	tau = (6.0f*visc + 1.0f)/2.0f;
	Re=(int)((mx-1)*0.1f/((2.0f*tau-1.0f)/6.0f)+0.5f);

	printf("visc = %f\n", visc);
	printf("tau = %f\n", tau);
	printf("Re = %d\n", Re);

	allocate();
	init();

//	hsize_t dimsf[] = {mx, my, 3};
//	hid_t file = H5Fcreate("out.dat", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
//	hid_t dataspace = H5Screate_simple(3, dimsf, NULL);
//	hid_t datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
//	H5Tset_order(datatype, H5T_ORDER_LE);

	for (st = 1; st < 10000; st++) {
		relaxate();
		propagate();
		if (st % 100 == 0) {
			output(st);	//, file, dataspace, datatype);
			printf("%05d\n", st);
		}
	}

//	H5Sclose(dataspace);
//	H5Tclose(datatype);
//	H5Fclose(file);

	return 0;
}
