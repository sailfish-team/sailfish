#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include "sim.h"
#include "vis.h"

struct Vis {
	int xres, yres, depth;
	int bytesPerPix, blockh, blockw;
	SDL_Surface *screen, *bbuf;
	SDL_Rect rbuf;
	TTF_Font *font;
} vis;

static SDL_Color text_color = { 255, 0, 0 };

void SDLDrawPixel(SDL_Surface *screen, int x, int y, u8 r, u8 g, u8 b)
{
	Uint32 color = SDL_MapRGB(screen->format, r, g, b);

	if (SDL_MUSTLOCK(screen)) {
		if (SDL_LockSurface(screen) < 0) {
			return;
		}
	}

	switch (screen->format->BytesPerPixel) {
	case 1: { /* Assuming 8-bpp */
		Uint8 *bufp;
		bufp = (Uint8 *)screen->pixels + y*screen->pitch + x;
		*bufp = color;
	}
	break;

	case 2: { /* Probably 15-bpp or 16-bpp */
		Uint16 *bufp;

		bufp = (Uint16 *)screen->pixels + y*screen->pitch/2 + x;
		*bufp = color;
	}
	break;

	case 3: { /* Slow 24-bpp mode, usually not used */
		Uint8 *bufp;

		bufp = (Uint8 *)screen->pixels + y*screen->pitch + x;
		*(bufp+screen->format->Rshift/8) = r;
		*(bufp+screen->format->Gshift/8) = g;
		*(bufp+screen->format->Bshift/8) = b;
	}
	break;

	case 4: { /* Probably 32-bpp */
		Uint32 *bufp;

		bufp = (Uint32 *)screen->pixels + y*screen->pitch/4 + x;
		*bufp = color;
	}
	break;
	}

	if (SDL_MUSTLOCK(screen)) {
		SDL_UnlockSurface(screen);
	}
	SDL_UpdateRect(screen, x, y, 1, 1);
}

Uint32 SDLGetPixel(SDL_Surface *surface, int x, int y)
{
	int bpp = surface->format->BytesPerPixel;
	/* Here p is the address to the pixel we want to retrieve */
	Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * bpp;

	switch(bpp) {
	case 1:
		return *p;

	case 2:
		return *(Uint16 *)p;

	case 3:
		if(SDL_BYTEORDER == SDL_BIG_ENDIAN)
			return p[0] << 16 | p[1] << 8 | p[2];
		else
			return p[0] | p[1] << 8 | p[2] << 16;

	case 4:
		return *(Uint32 *)p;

	default:
		return 0;       /* shouldn't happen, but avoids warnings */
	}
}

void SDLPutPixel(SDL_Surface *surface, int x, int y, Uint32 pixel)
{
	int bpp = surface->format->BytesPerPixel;
	/* Here p is the address to the pixel we want to set */
	Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * bpp;

	switch(bpp) {
	case 1:
		*p = pixel;
		 break;

	case 2:
		*(Uint16 *)p = pixel;
		break;

	case 3:
		if(SDL_BYTEORDER == SDL_BIG_ENDIAN) {
			p[0] = (pixel >> 16) & 0xff;
			p[1] = (pixel >> 8) & 0xff;
			p[2] = pixel & 0xff;
		} else {
			p[0] = pixel & 0xff;
			p[1] = (pixel >> 8) & 0xff;
			p[2] = (pixel >> 16) & 0xff;
		}
		break;

	case 4:
		*(Uint32 *)p = pixel;
		break;
    }
}

void SDLDrawBlockPixel(SDL_Surface *screen, int x, int y, u8 r, u8 g, u8 b)
{
	Uint32 color = SDL_MapRGB(screen->format, r, g, b);

	if (SDL_MUSTLOCK(screen)) {
		if (SDL_LockSurface(screen) < 0) {
			return;
		}
	}

	x = x * VIS_BLOCK_SIZE;
	y = y * VIS_BLOCK_SIZE;
	int i, j;

	switch (screen->format->BytesPerPixel) {
	case 1: { /* Assuming 8-bpp */
		Uint8 *bufp;
		bufp = (Uint8 *)screen->pixels + y*screen->pitch + x;

		for (j = 0; j < VIS_BLOCK_SIZE; j++) {
			for (i = 0; i < VIS_BLOCK_SIZE; i++) {
				*(bufp+i) = color;
			}
			bufp += screen->pitch;
		}
	}
	break;

	case 2: { /* Probably 15-bpp or 16-bpp */
		Uint16 *bufp;

		bufp = (Uint16 *)screen->pixels + y*screen->pitch/2 + x;
		for (j = 0; j < VIS_BLOCK_SIZE; j++) {
			for (i = 0; i < VIS_BLOCK_SIZE; i++) {
				*(bufp+i) = color;
			}
			bufp += screen->pitch/2;
		}
	}
	break;

	case 3: { /* Slow 24-bpp mode, usually not used */
		for (j = 0; j < VIS_BLOCK_SIZE; j++) {
			for (i = 0; i < VIS_BLOCK_SIZE; i++) {
				SDLPutPixel(screen, x + i, y + j, color);
			}
		}
	}
	break;

	case 4: { /* Probably 32-bpp */
		Uint32 *bufp;

		bufp = (Uint32 *)screen->pixels + y*screen->pitch/4 + x;
		for (j = 0; j < VIS_BLOCK_SIZE; j++) {
			for (i = 0; i < VIS_BLOCK_SIZE; i++) {
				*(bufp+i) = color;
			}
			bufp += screen->pitch/4;
		}
	}
	break;
	}

	if (SDL_MUSTLOCK(screen)) {
		SDL_UnlockSurface(screen);
	}
	SDL_UpdateRect(screen, x, y, VIS_BLOCK_SIZE, VIS_BLOCK_SIZE);
}

void render_text(SDL_Surface *dst, char *text, int x, int y)
{
	SDL_Surface *tmp;
	SDL_Rect dstrect;

	tmp = TTF_RenderText_Solid(vis.font, text, text_color);
	if (tmp != NULL) {
		dstrect.x = x;
		dstrect.y = y;
		dstrect.w = tmp->w;
		dstrect.h = tmp->h;
		SDL_BlitSurface(tmp, NULL, dst, &dstrect);
		SDL_FreeSurface(tmp);
	}
}

void visualize(struct SimState *state, float lups)
{
	int x, y, i = 0;
	char buf[128];

	for (y = 0; y < LAT_H; y++) {
		for (x = 0; x < LAT_W; x++) {
			float amnt = sqrtf(state->vx[i]*state->vx[i] + state->vy[i]*state->vy[i]) / 0.1f * 255.0f;
			if (state->map[i] == GEO_WALL) {
				SDLDrawBlockPixel(vis.bbuf, x, LAT_H-y-1, 0, 0, 255);
			} else {
				SDLDrawBlockPixel(vis.bbuf, x, LAT_H-y-1, (int)amnt, (int)amnt, 0);
			}
			i++;
		}
	}
	sprintf(buf, "%.1f MLUPS", lups * 1e-6);
	render_text(vis.bbuf, buf, 0, VIS_BLOCK_SIZE * LAT_H - 16);

	SDL_BlitSurface(vis.bbuf, NULL, vis.screen, &vis.rbuf);
	SDL_UpdateRect(vis.screen, 0, 0, 0, 0);
}

void SDLInit(void)
{
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
		exit(1);

	if (TTF_Init() < 0)
		exit(2);

	vis.font = TTF_OpenFont("lucida.ttf", 12);
	if (vis.font == NULL) {
		fprintf(stderr, "Couldn't load %d pt font from %s: %s\n", 12, "lucida.ttf", SDL_GetError());
		exit(3);
	}

	TTF_SetFontStyle(vis.font, TTF_STYLE_NORMAL);
	vis.xres = LAT_W*VIS_BLOCK_SIZE;
	vis.yres = LAT_H*VIS_BLOCK_SIZE;
	vis.depth = 32;

	if (!(vis.screen = SDL_SetVideoMode(vis.xres, vis.yres, vis.depth, SDL_HWSURFACE))) {
		SDL_Quit();
		exit(1);
	}

	vis.bytesPerPix = vis.screen->format->BytesPerPixel;
//	vis.blockw = vis.screen->pitch/bytesPerPix/BLOCK;
//	vis.blockh = vis.screen->h/BLOCK;

	vis.bbuf = SDL_CreateRGBSurface(SDL_HWSURFACE, vis.screen->w, vis.screen->h, vis.screen->format->BitsPerPixel,
									vis.screen->format->Rmask, vis.screen->format->Gmask, vis.screen->format->Bmask, vis.screen->format->Amask);

	vis.rbuf.x = 0;
	vis.rbuf.y = 0;
	vis.rbuf.w = vis.bbuf->w;
	vis.rbuf.h = vis.bbuf->h;
}

