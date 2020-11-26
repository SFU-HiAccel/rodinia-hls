#ifndef KNN_H
#define KNN_H

// #define NAIVE_0
// #define TILING_1
// #define PIPELINING_2
// #define UNROLLING_3
// #define DBLBUFFERING_4
//#define COALESCING_5_512bit
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

const int NUM_FEATURE = 2;
const int NUM_PT_IN_SEARCHSPACE = 1024*1024;
const int NUM_PT_IN_BUFFER = 512;
const int NUM_TILES = NUM_PT_IN_SEARCHSPACE / NUM_PT_IN_BUFFER;
const int UNROLL_FACTOR = 2;

#ifdef COALESCING_5_512bit
#include <gmp.h>
#define __gmp_const const
#include "ap_int.h"
#include <inttypes.h>
    const int DWIDTH = 512;
    #define INTERFACE_WIDTH ap_uint<DWIDTH>
    const int WIDTH_FACTOR = DWIDTH/32;
#endif 

// Definition for testbench
// void workload(
//     float inputQuery[NUM_FEATURE],
//     float searchSpace[NUM_PT_IN_SEARCHSPACE*NUM_FEATURE],
//     float distance[NUM_PT_IN_SEARCHSPACE]
// );

struct bench_args_t {
    float input_query[NUM_FEATURE];
    float search_space_data[NUM_PT_IN_SEARCHSPACE*NUM_FEATURE];
    float distance[NUM_PT_IN_SEARCHSPACE];
};

#endif
