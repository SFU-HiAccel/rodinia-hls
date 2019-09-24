#ifndef HOTSPOT_H
#define HOTSPOT_H


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>


#define SIM_TIME 64

#define GRID_ROWS 512
#define GRID_COLS 512

#define TILE_ROWS 64

#define PARA_FACTOR 16



/* maximum power density possible (say 300W for a 10mm x 10mm chip) */

#define MAX_PD  (3.0e4)

/* required precision in degrees  */
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100

/* capacitance fitting factor */
#define FACTOR_CHIP 0.5
#define OPEN

/* chip parameters  */
#define T_CHIP 0.0005
#define CHIP_HEIGHT 0.016
#define CHIP_WIDTH 0.016

#define AMB_TEMP 80.0

#define TOP 0
#define BOTTOM (GRID_ROWS/TILE_ROWS - 1)

#define TYPE float


struct bench_args_t {
    float temp[GRID_ROWS * GRID_COLS];
    float power[GRID_ROWS * GRID_COLS];
};


#endif
