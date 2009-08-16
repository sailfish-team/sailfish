// The following additional constants need to be defined:
// LAT_H, LAT_W, BLOCK_SIZE, GEO_FLUID, GEO_WALL, GEO_BCV, GEO_BCP
// NUM_PARAMS

#define RELAX_bgk	BGK_relaxate(rho, v, fi, map[gi]);
#define RELAX_mrt	MS_relaxate(fi, map[gi]);

#define DT 1.0f

__constant__ float tau;			// relaxation time
__constant__ float visc;		// viscosity
__constant__ float geo_params[NUM_PARAMS];		// geometry parameters

struct DistP {
	float *fC, *fE, *fW, *fS, *fN, *fSE, *fSW, *fNE, *fNW;
};

struct Dist {
	float fC, fE, fW, fS, fN, fSE, fSW, fNE, fNW;
};

// Distribution in momentum space.
struct DistM {
	float rho, en, ens, mx, ex, my, ey, sd, sod;
};

//
// Copy the idx-th distribution from din into dout.
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
// the node class node_type.
//
__device__ void inline getMacro(Dist fi, int node_type, float &rho, float2 &v)
{
	rho = fi.fC + fi.fE + fi.fW + fi.fS + fi.fN + fi.fNE + fi.fNW + fi.fSE + fi.fSW;
	if (node_type >= GEO_BCV) {
		// Velocity boundary condition.
		if (node_type < GEO_BCP) {
			int idx = (node_type - GEO_BCV) * 2;
			v.x = geo_params[idx];
			v.y = geo_params[idx+1];
			return;
		// Pressure boundary condition.
		} else {
			// c_s^2 = 1/3, P/c_s^2 = rho
			int idx = (GEO_BCP-GEO_BCV) * 2 + (node_type - GEO_BCP);
			rho = geo_params[idx] * 3.0;
		}
	}

	v.x = (fi.fE + fi.fSE + fi.fNE - fi.fW - fi.fSW - fi.fNW) / rho + 0.5f * ext_accel_x;
	v.y = (fi.fN + fi.fNW + fi.fNE - fi.fS - fi.fSW - fi.fSE) / rho + 0.5f * ext_accel_y;
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
// Relaxation in moment space.
//
__device__ void inline MS_relaxate(Dist &fi, int node_type)
{
	DistM fm, feq;

	fm.rho = 1.0f*fi.fC + 1.0f*fi.fE + 1.0f*fi.fN + 1.0f*fi.fW + 1.0f*fi.fS + 1.0f*fi.fNE + 1.0f*fi.fNW + 1.0f*fi.fSW + 1.0f*fi.fSE;
	fm.en = -4.0f*fi.fC - 1.0f*fi.fE - 1.0f*fi.fN - 1.0f*fi.fW - 1.0f*fi.fS + 2.0f*fi.fNE + 2.0f*fi.fNW + 2.0f*fi.fSW + 2.0f*fi.fSE;
	fm.ens = 4.0f*fi.fC - 2.0f*fi.fE - 2.0f*fi.fN - 2.0f*fi.fW - 2.0f*fi.fS + 1.0f*fi.fNE + 1.0f*fi.fNW + 1.0f*fi.fSW + 1.0f*fi.fSE;
	fm.mx =  0.0f*fi.fC + 1.0f*fi.fE + 0.0f*fi.fN - 1.0f*fi.fW + 0.0f*fi.fS + 1.0f*fi.fNE - 1.0f*fi.fNW - 1.0f*fi.fSW + 1.0f*fi.fSE;
	fm.ex =  0.0f*fi.fC - 2.0f*fi.fE + 0.0f*fi.fN + 2.0f*fi.fW + 0.0f*fi.fS + 1.0f*fi.fNE - 1.0f*fi.fNW - 1.0f*fi.fSW + 1.0f*fi.fSE;
	fm.my =  0.0f*fi.fC + 0.0f*fi.fE + 1.0f*fi.fN + 0.0f*fi.fW - 1.0f*fi.fS + 1.0f*fi.fNE + 1.0f*fi.fNW - 1.0f*fi.fSW - 1.0f*fi.fSE;
	fm.ey =  0.0f*fi.fC + 0.0f*fi.fE - 2.0f*fi.fN + 0.0f*fi.fW + 2.0f*fi.fS + 1.0f*fi.fNE + 1.0f*fi.fNW - 1.0f*fi.fSW - 1.0f*fi.fSE;
	fm.sd =  0.0f*fi.fC + 1.0f*fi.fE - 1.0f*fi.fN + 1.0f*fi.fW - 1.0f*fi.fS + 0.0f*fi.fNE + 0.0f*fi.fNW + 0.0f*fi.fSW - 0.0f*fi.fSE;
	fm.sod = 0.0f*fi.fC + 0.0f*fi.fE + 0.0f*fi.fN + 0.0f*fi.fW + 0.0f*fi.fS + 1.0f*fi.fNE - 1.0f*fi.fNW + 1.0f*fi.fSW - 1.0f*fi.fSE;

	if (node_type >= GEO_BCV) {
		// Velocity boundary condition.
		if (node_type < GEO_BCP) {
			int idx = (node_type - GEO_BCV) * 2;
			fm.mx = geo_params[idx];
			fm.my = geo_params[idx+1];
		// Pressure boundary condition.
		} else {
			int idx = (GEO_BCP-GEO_BCV) * 2 + (node_type - GEO_BCP);
			fm.rho = geo_params[idx] * 3.0f;
		}
	}

	float h = fm.mx*fm.mx + fm.my*fm.my;
	feq.en  = -2.0f*fm.rho + 3.0f*h;
	feq.ens = fm.rho - 3.0f*h;
	feq.ex  = -fm.mx;
	feq.ey  = -fm.my;
	feq.sd  = (fm.mx*fm.mx - fm.my*fm.my);
	feq.sod = (fm.mx*fm.my);

	float tau7 = 4.0f / (12.0f*visc + 2.0f);
	float tau4 = 3.0f*(2.0f - tau7) / (3.0f - tau7);
	float tau8 = 1.0f/((2.0f/tau7 - 1.0f)*0.5f + 0.5f);

	if (node_type == GEO_FLUID || node_type == GEO_WALL) {
		fm.en  -= 1.63f * (fm.en - feq.en);
		fm.ens -= 1.14f * (fm.ens - feq.ens);
		fm.ex  -= tau4 * (fm.ex - feq.ex);
		fm.ey  -= 1.92f * (fm.ey - feq.ey);
		fm.sd  -= tau7 * (fm.sd - feq.sd);
		fm.sod -= tau8 * (fm.sod - feq.sod);
	} else {
		fm.en  = feq.en;
		fm.ens = feq.ens;
		fm.ex  = feq.ex;
		fm.ey  = feq.ey;
		fm.sd  = feq.sd;
		fm.sod = feq.sod;
	}

	fi.fC  = (1.0f/9.0f)*fm.rho - (1.0f/9.0f)*fm.en + (1.0f/9.0f)*fm.ens;
	fi.fE  = (1.0f/9.0f)*fm.rho - (1.0f/36.0f)*fm.en - (1.0f/18.0f)*fm.ens + (1.0f/6.0f)*fm.mx - (1.0f/6.0f)*fm.ex + 0.25f*fm.sd;
	fi.fN  = (1.0f/9.0f)*fm.rho - (1.0f/36.0f)*fm.en - (1.0f/18.0f)*fm.ens + (1.0f/6.0f)*fm.my - (1.0f/6.0f)*fm.ey - 0.25f*fm.sd;
	fi.fW  = (1.0f/9.0f)*fm.rho - (1.0f/36.0f)*fm.en - (1.0f/18.0f)*fm.ens - (1.0f/6.0f)*fm.mx + (1.0f/6.0f)*fm.ex + 0.25f*fm.sd;
	fi.fS  = (1.0f/9.0f)*fm.rho - (1.0f/36.0f)*fm.en - (1.0f/18.0f)*fm.ens - (1.0f/6.0f)*fm.my + (1.0f/6.0f)*fm.ey - 0.25f*fm.sd;
	fi.fNE = (1.0f/9.0f)*fm.rho + (1.0f/18.0f)*fm.en + (1.0f/36.0f)*fm.ens +
			 +(1.0f/6.0f)*fm.mx + (1.0f/12.0f)*fm.ex + (1.0f/6.0f)*fm.my + (1.0f/12.0f)*fm.ey + 0.25f*fm.sod;
	fi.fNW = (1.0f/9.0f)*fm.rho + (1.0f/18.0f)*fm.en + (1.0f/36.0f)*fm.ens +
			 -(1.0f/6.0f)*fm.mx - (1.0f/12.0f)*fm.ex + (1.0f/6.0f)*fm.my + (1.0f/12.0f)*fm.ey - 0.25f*fm.sod;
	fi.fSW = (1.0f/9.0f)*fm.rho + (1.0f/18.0f)*fm.en + (1.0f/36.0f)*fm.ens +
			 -(1.0f/6.0f)*fm.mx - (1.0f/12.0f)*fm.ex - (1.0f/6.0f)*fm.my - (1.0f/12.0f)*fm.ey + 0.25f*fm.sod;
	fi.fSE = (1.0f/9.0f)*fm.rho + (1.0f/18.0f)*fm.en + (1.0f/36.0f)*fm.ens +
			 +(1.0f/6.0f)*fm.mx + (1.0f/12.0f)*fm.ex - (1.0f/6.0f)*fm.my - (1.0f/12.0f)*fm.ey - 0.25f*fm.sod;
}

//
// Performs the relaxation step in the BGK model given the density rho,
// the velocity v and the distribution fi.
//
__device__ void inline BGK_relaxate(float rho, float2 v, Dist &fi, int node_type)
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

 	if (node_type == GEO_FLUID || node_type == GEO_WALL) {
		fi.fC += (feq.fC - fi.fC) / tau;
		fi.fE += (feq.fE - fi.fE) / tau;
		fi.fW += (feq.fW - fi.fW) / tau;
		fi.fS += (feq.fS - fi.fS) / tau;
		fi.fN += (feq.fN - fi.fN) / tau;
		fi.fSE += (feq.fSE - fi.fSE) / tau;
		fi.fNE += (feq.fNE - fi.fNE) / tau;
		fi.fSW += (feq.fSW - fi.fSW) / tau;
		fi.fNW += (feq.fNW - fi.fNW) / tau;
	} else {
		fi.fC  = feq.fC;
		fi.fE  = feq.fE;
		fi.fW  = feq.fW;
		fi.fS  = feq.fS;
		fi.fN  = feq.fN;
		fi.fSE = feq.fSE;
		fi.fNE = feq.fNE;
		fi.fSW = feq.fSW;
		fi.fNW = feq.fNW;
	}

	float pref = rho * (3.0f - 3.0f/(2.0f * tau));
	#define eax ext_accel_x
	#define eay ext_accel_y
	float ue = eax*v.x + eay*v.y;

	fi.fC += pref*(-ue) * 4.0f/9.0f;
	fi.fN += pref*(eay - ue + 3.0f*(eay*v.y)) / 9.0f;
	fi.fE += pref*(eax - ue + 3.0f*(eax*v.x)) / 9.0f;
	fi.fS += pref*(-eay - ue + 3.0f*(eay*v.y)) / 9.0f;
	fi.fW += pref*(-eax - ue + 3.0f*(eax*v.x)) / 9.0f;
	fi.fNE += pref*(eax + eay - ue + 3.0f*((v.x+v.y)*(eax+eay))) / 36.0f;
	fi.fSE += pref*(eax - eay - ue + 3.0f*((v.x-v.y)*(eax-eay))) / 36.0f;
	fi.fSW += pref*(-eax - eay - ue + 3.0f*((v.x+v.y)*(eax+eay))) / 36.0f;
	fi.fNW += pref*(-eax + eay - ue + 3.0f*((-v.x+v.y)*(-eax+eay))) / 36.0f;
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

	int type = map[gi];

	if (type == GEO_WALL) {
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

	// macroscopic quantities for the current cell
	float rho;
	float2 v;
	getMacro(fi, type, rho, v);

	// only save the macroscopic quantities if requested to do so
	if (orho != NULL) {
		orho[gi] = rho;
		ovx[gi] = v.x;
		ovy[gi] = v.y;
	}

	RELAXATE;

	// update the 0-th direction distribution
	od.fC[gi] = fi.fC;

	#define set_odist(idx, dir, val) od.dir[idx] = val
	#define rel(x,y) ((x) + LAT_W*(y))

	// E propagation in shared memory
	if (tix < blockDim.x-1) {
		fo_E[tix+1] = fi.fE;
		fo_NE[tix+1] = fi.fNE;
		fo_SE[tix+1] = fi.fSE;
	// E propagation in global memory (at right block boundary)
	} else if (ti < LAT_W-1) {
		set_odist(gi+rel(1,0), fE, fi.fE);
		if (blockIdx.y > 0)			set_odist(gi+rel(1,-1), fSE, fi.fSE);
		else if (PERIODIC_Y)		set_odist(ti+LAT_W*(LAT_H-1)+1, fSE, fi.fSE);
		if (blockIdx.y < LAT_H-1)	set_odist(gi+rel(1,1), fNE, fi.fNE);
		else if (PERIODIC_Y)		set_odist(ti+1, fNE, fi.fNE);
	} else if (PERIODIC_X) {
		set_odist(gi+rel(-LAT_W+1, 0), fE, fi.fE);
		if (blockIdx.y > 0)			set_odist(gi+rel(-LAT_W+1,-1), fSE, fi.fSE);
		else if (PERIODIC_Y)		set_odist(rel(0, LAT_H-1), fSE, fi.fSE);
		if (blockIdx.y < LAT_H-1)	set_odist(gi+rel(-LAT_W+1,1), fNE, fi.fNE);
		else if (PERIODIC_Y)		set_odist(rel(0, 0), fNE, fi.fNE);
	}

	// W propagation in shared memory
	if (tix > 0) {
		fo_W[tix-1] = fi.fW;
		fo_NW[tix-1] = fi.fNW;
		fo_SW[tix-1] = fi.fSW;
	// W propagation in global memory (at left block boundary)
	} else if (ti > 0) {
		set_odist(gi+rel(-1,0), fW, fi.fW);
		if (blockIdx.y > 0)			set_odist(gi+rel(-1,-1), fSW, fi.fSW);
		else if (PERIODIC_Y)		set_odist(ti+LAT_W*(LAT_H-1)-1, fSW, fi.fSW);
		if (blockIdx.y < LAT_H-1)	set_odist(gi+rel(-1,1), fNW, fi.fNW);
		else if (PERIODIC_Y)		set_odist(ti-1, fNW, fi.fNW);
	} else if (PERIODIC_X) {
		set_odist(gi+rel(LAT_W-1,0), fW, fi.fW);
		if (blockIdx.y > 0)			set_odist(gi+rel(LAT_W-1,-1), fSW, fi.fSW);
		else if (PERIODIC_Y)		set_odist(LAT_H*LAT_W-1, fSW, fi.fSW);
		if (blockIdx.y < LAT_H-1)	set_odist(gi+rel(LAT_W-1,1), fNW, fi.fNW);
		else if (PERIODIC_Y)		set_odist(LAT_W-1, fNW, fi.fNW);
	}

	__syncthreads();

	// the leftmost thread is not updated in this block
	if (tix > 0) {
		set_odist(gi, fE, fo_E[tix]);
		if (blockIdx.y > 0)			set_odist(gi-LAT_W, fSE, fo_SE[tix]);
		else if (PERIODIC_Y)		set_odist(gi+LAT_W*(LAT_H-1), fSE, fo_SE[tix]);
		if (blockIdx.y < LAT_H-1)	set_odist(gi+LAT_W, fNE, fo_NE[tix]);
		else if (PERIODIC_Y)		set_odist(ti, fNE, fo_NE[tix]);
	}

	// N + S propagation (global memory)
	if (blockIdx.y > 0)			set_odist(gi-LAT_W, fS, fi.fS);
	else if (PERIODIC_Y)		set_odist(ti+LAT_W*(LAT_H-1), fS, fi.fS);
	if (blockIdx.y < LAT_H-1)	set_odist(gi+LAT_W, fN, fi.fN);
	else if (PERIODIC_Y)		set_odist(ti, fN, fi.fN);

	// the rightmost thread is not updated in this block
	if (tix < blockDim.x-1) {
		set_odist(gi, fW, fo_W[tix]);
		if (blockIdx.y > 0)			set_odist(gi-LAT_W, fSW, fo_SW[tix]);
		else if (PERIODIC_Y)		set_odist(gi+LAT_W*(LAT_H-1), fSW, fo_SW[tix]);
		if (blockIdx.y < LAT_H-1)	set_odist(gi+LAT_W, fNW, fo_NW[tix]);
		else if (PERIODIC_Y)		set_odist(ti, fNW, fo_NW[tix]);
	}
}

