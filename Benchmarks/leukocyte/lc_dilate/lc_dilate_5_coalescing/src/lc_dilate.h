#ifndef LC_DILATE_H
#define LC_DILATE_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#define GRID_ROWS 512
#define GRID_COLS 512

#define TILE_ROWS 64

#define PARA_FACTOR 32

#define STREL_ROWS 5
#define STREL_COLS 5

#define MAX_RADIUS 2



#define PI 3.1415926
#define TOP 0
#define BOTTOM (GRID_ROWS / TILE_ROWS - 1)



#define TYPE float

struct bench_args_t {
    float result[GRID_ROWS * GRID_COLS];
    float img[GRID_ROWS * GRID_COLS];
};


#endif
