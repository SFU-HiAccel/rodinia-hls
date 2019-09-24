#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//-----------------------------------------------
//Original
// #define ROWS 16384
// #define COLS 16384
//-----------------------------------------------
//-----------------------------------------------
//Alec-added
#define ROWS 1024
#define COLS 1024
//-----------------------------------------------
#define TILE_SIZE 16
#define TYPE int32_t
#define MIN(a,b) ((a)<=(b) ? (a) : (b))

void pathfinder_kernel(int32_t J[ROWS*COLS], int32_t Jout[COLS]);

////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  int32_t J[ROWS*COLS];
  int32_t Jout[COLS];
};
