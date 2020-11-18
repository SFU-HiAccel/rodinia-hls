#ifndef CFD_FLUX_H
#define CFD_FLUX_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GRID_ROWS (1024)

#define SIZE GRID_ROWS

#define TILE_ROWS 32

#define PARA_FACTOR 16

#define TOP 0
#define BOTTOM (GRID_ROWS / TILE_ROWS - 1)



#define GAMMA 1.4

#define NDIM 3
#define NNB 4

#define RK 3	// 3rd order RK
#define ff_mach 1.2
#define deg_angle_of_attack 0.0f

#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)


#define TYPE float

struct float3 { float x, y, z; };


struct bench_args_t {

    float result                          [SIZE *              NVAR];
    float elements_surrounding_elements   [SIZE * NNB];
    float normals                         [SIZE * NNB * NDIM];
    float variables                       [SIZE *              NVAR];
    
    float fc_momentum_x                   [SIZE *       NDIM];
    float fc_momentum_y                   [SIZE *       NDIM];
    float fc_momentum_z                   [SIZE *       NDIM];
    float fc_density_energy               [SIZE *       NDIM];

};


#endif
