#ifndef __VIS_H_
#define __VIS_H_ 1

#include <SDL/SDL.h>

typedef char u8;

// forward declaration
struct SimState;

#define VIS_BLOCK_SIZE 3

void SDLDrawPixel(SDL_Surface *screen, int x, int y, u8 r, u8 g, u8 b);
Uint32 SDLGetPixel(SDL_Surface *surface, int x, int y);
void SDLPutPixel(SDL_Surface *surface, int x, int y, Uint32 pixel);
void SDLDrawBlockPixel(SDL_Surface *screen, int x, int y, u8 r, u8 g, u8 b);
void visualize(struct SimState *state);
void SDLInit(void);

#endif  /* __VIS_H_ */
