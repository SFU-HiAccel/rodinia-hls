#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TILE_ROWS 16
#define ROWS 1024
#define COLS 1024
#define R1 0
#define R2 63
#define C1 0
#define C2 63
#define LAMBDA 0.5 
#define PARA_FACTOR 16
#define NITER 2
#define TYPE float
#define TOP_TILE 0
#define BOTTOM_TILE (ROWS/TILE_ROWS - 1)

float srad_core1(float dN, float dS, float dW, float dE,
		  float Jc, float q0sqr);
float srad_core2 (float dN, float dS, float dW, float dE,
		  float cN, float cS, float cW, float cE,
		  float J);
//void srad_kernel1(float J[(ROWS+3)*COLS], float q0sqr[1]);
void srad_kernel2(float J[(TILE_ROWS+3)*COLS], float Jout[TILE_ROWS*COLS], float q0sqr, int tile);

////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  float J[(ROWS+3)*COLS];
  float Jout[(ROWS+3)*COLS];
};
