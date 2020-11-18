#ifndef KMEANS_H
#define KMEANS_H

#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../../../common/mc.h"

#define FLT_MAX 3.40282347e+38

#define NPOINTS (819200/2)
#define NFEATURES 34
#define NCLUSTERS 5
#define TILE_SIZE 4096

const int WIDTH_FACTOR = 16;

const int NUM_TILES = NPOINTS/TILE_SIZE;

// void workload(float  *feature, /* [npoints][nfeatures] */
// 			  float  *clusters, /* [n_clusters][n_features] */
// 			  int *membership);

struct bench_args_t {
	float FEATURE[NPOINTS*NFEATURES];
	float CLUSTER[NCLUSTERS*NFEATURES];
	int MEMBERSHIP[NPOINTS];
};

#endif
