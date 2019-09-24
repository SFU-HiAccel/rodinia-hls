#ifndef LC_MGVF_H
#define LC_MGVF_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#define GRID_ROWS 1024
#define GRID_COLS 1024

#define TILE_ROWS 64

#define PARA_FACTOR 16


#define ITERATION 64


#define TOP 0
#define BOTTOM (GRID_ROWS / TILE_ROWS - 1)


#define MU 0.5
#define LAMBDA (8.0 * MU + 1.0)
#define MU_O_LAMBDA  (MU / LAMBDA)
#define ONE_O_LAMBDA (1.0 / LAMBDA)


#define COVERAGE 0.000000000001
#define MAX_RADIUS 1

#define TYPE float

struct bench_args_t {
    float imgvf[GRID_ROWS * GRID_COLS];
    float I[GRID_ROWS * GRID_COLS];
};




#endif
