#define __kernel
#define __global
#include <math.h>
#include <string.h>
#ifndef LARGE_BUS
#define LARGE_BUS 512
#endif
#define MARS_WIDE_BUS_TYPE ap_uint<LARGE_BUS>
#include "ap_int.h"
#include "../../../../common/mars_wide_bus.h"

#define ETA 0.3       //eta value
#define MOMENTUM 0.3  //momentum value
#define TILE_SIZE 4096

extern "C" {
    
void buffer_load(int flag, float *w, float w_buf[TILE_SIZE][16], float *oldw, float oldw_buf[TILE_SIZE][16]) {
#pragma HLS INLINE off
    if(flag) {
        memcpy(w_buf[0], w, sizeof(float) * TILE_SIZE * 16);
        memcpy(oldw_buf[0], oldw, sizeof(float) * TILE_SIZE * 16);
    }
}

void buffer_store(int flag, float *w, float w_buf[TILE_SIZE][16], float *oldw, float oldw_buf[TILE_SIZE][16]) {
#pragma HLS INLINE off
    if(flag) {
        memcpy(w, w_buf[0], sizeof(float) * TILE_SIZE * 16);
        memcpy(oldw, oldw_buf[0], sizeof(float) * TILE_SIZE * 16);
    }
}

void buffer_compute(int flag, int kk, float delta_buf[16], float ly_buf[65536], float w_buf[TILE_SIZE][16], float oldw_buf[TILE_SIZE][16]) {
#pragma HLS INLINE off
    float new_dw;
    int j, k;
    if (flag) {
        LOOP2:for (k = 0; k < TILE_SIZE; k++) {
        #pragma HLS PIPELINE
            LOOP3: for (j = 0; j < 16; j++) {
            #pragma HLS UNROLL
                new_dw = ETA * delta_buf[j] * ly_buf[k+kk] + MOMENTUM * oldw_buf[k][j];
                w_buf[k][j] += new_dw;
                oldw_buf[k][j] = new_dw;
            }
        }
    }
}

void workload(float delta[17], float ly[65537], float w[65537 * 16], float oldw[65537 * 16]) {
#pragma HLS INTERFACE m_axi port=delta offset=slave bundle=delta
#pragma HLS INTERFACE m_axi port=ly offset=slave bundle=ly
#pragma HLS INTERFACE m_axi port=w offset=slave bundle=w
#pragma HLS INTERFACE m_axi port=oldw offset=slave bundle=oldw

#pragma HLS INTERFACE s_axilite port=delta bundle=control
#pragma HLS INTERFACE s_axilite port=ly bundle=control
#pragma HLS INTERFACE s_axilite port=w bundle=control
#pragma HLS INTERFACE s_axilite port=oldw bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    
    float new_dw;
    int kk, i, j;

    float w0_buf[16];
    #pragma HLS ARRAY_PARTITION variable=w0_buf complete dim=1
    float oldw0_buf[16];
    #pragma HLS ARRAY_PARTITION variable=oldw0_buf complete dim=1
    float delta_buf[16];
    #pragma HLS ARRAY_PARTITION variable=delta_buf complete dim=1

    float ly_buf[65536];

    float w_buf_x[TILE_SIZE][16];
    #pragma HLS ARRAY_PARTITION variable=w_buf_x complete dim=2

    float w_buf_y[TILE_SIZE][16];
    #pragma HLS ARRAY_PARTITION variable=w_buf_y complete dim=2

    float w_buf_z[TILE_SIZE][16];
    #pragma HLS ARRAY_PARTITION variable=w_buf_z complete dim=2

    float oldw_buf_x[TILE_SIZE][16];
    #pragma HLS ARRAY_PARTITION variable=oldw_buf_x complete dim=2

    float oldw_buf_y[TILE_SIZE][16];
    #pragma HLS ARRAY_PARTITION variable=oldw_buf_y complete dim=2

    float oldw_buf_z[TILE_SIZE][16];
    #pragma HLS ARRAY_PARTITION variable=oldw_buf_z complete dim=2

    memcpy(delta_buf, delta+1, sizeof(float)*16);
    memcpy(ly_buf, ly+1, sizeof(float)*65536);
    memcpy(w0_buf, w, sizeof(float)*16);
    memcpy(oldw0_buf, oldw, sizeof(float)*16);
    
    LOOP1: for (kk = 0; kk < 65536+2*TILE_SIZE; kk+=TILE_SIZE) {
        int load_flag = ( kk < 65536 );
        int compute_flag = (( kk >= TILE_SIZE ) && ( kk < 65536+TILE_SIZE ));
        int store_flag = ( kk >= 2 * TILE_SIZE );
        i = (kk/TILE_SIZE)%3;

        if (i == 0) {
            buffer_load(load_flag, w+(kk+1)*16, w_buf_x, oldw+(kk+1)*16, oldw_buf_x);
            buffer_compute(compute_flag, kk, delta_buf, ly_buf, w_buf_z, oldw_buf_z);
            buffer_store(store_flag, w+(kk+1)*16-2*TILE_SIZE*16, w_buf_y, oldw+(kk+1)*16-2*TILE_SIZE*16, oldw_buf_y);
        }
        else if (i == 1) {
            buffer_load(load_flag, w+(kk+1)*16, w_buf_y, oldw+(kk+1)*16, oldw_buf_y);
            buffer_compute(compute_flag, kk, delta_buf, ly_buf, w_buf_x, oldw_buf_x);
            buffer_store(store_flag, w+(kk+1)*16-2*TILE_SIZE*16, w_buf_z, oldw+(kk+1)*16-2*TILE_SIZE*16, oldw_buf_z);
        }
        else {
            buffer_load(load_flag, w+(kk+1)*16, w_buf_z, oldw+(kk+1)*16, oldw_buf_z);
            buffer_compute(compute_flag, kk, delta_buf, ly_buf, w_buf_y, oldw_buf_y);
            buffer_store(store_flag, w+(kk+1)*16-2*TILE_SIZE*16, w_buf_x, oldw+(kk+1)*16-2*TILE_SIZE*16, oldw_buf_x);
        }
    }

    LOOP4: for (j = 0; j < 16; j++) {
    #pragma HLS UNROLL
        new_dw = ETA * delta_buf[j] + MOMENTUM * oldw0_buf[j];
        w0_buf[j] += new_dw;
        oldw0_buf[j] = new_dw;
    }

    memcpy(w, w0_buf, sizeof(float)*16);
    memcpy(oldw, oldw0_buf, sizeof(float)*16);
}

}