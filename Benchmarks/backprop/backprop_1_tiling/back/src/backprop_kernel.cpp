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
    int k, kk, kkk, j;

    float w0_buf[16];
    float oldw0_buf[16];

    float delta_buf[16];

    float ly_buf[65536];

    float w_buf[TILE_SIZE][16];
    float oldw_buf[TILE_SIZE][16];


    memcpy(delta_buf, delta+1, sizeof(float)*16);
    memcpy(ly_buf, ly+1, sizeof(float)*65536);
    memcpy(w0_buf, w, sizeof(float)*16);
    memcpy(oldw0_buf, oldw, sizeof(float)*16);
    
    LOOP1: for (kk = 1; kk < 65537; kk+=TILE_SIZE) {
        memcpy(w_buf[0], w+kk*16, sizeof(float) * TILE_SIZE * 16);
        memcpy(oldw_buf[0], oldw+kk*16, sizeof(float) * TILE_SIZE * 16);
        LOOP2:for (k = 0; k < TILE_SIZE; k++) {
            LOOP3: for (j = 0; j < 16; j++) {
                new_dw = ETA * delta_buf[j] * ly_buf[k+kk-1] + MOMENTUM * oldw_buf[k][j];
                w_buf[k][j] += new_dw;
                oldw_buf[k][j] = new_dw;
            }
        }
        memcpy(w+kk*16, w_buf[0], sizeof(float) * TILE_SIZE * 16);
        memcpy(oldw+kk*16, oldw_buf[0], sizeof(float) * TILE_SIZE * 16);
    }

    LOOP4: for (j = 0; j < 16; j++) {
        new_dw = ETA * delta_buf[j] + MOMENTUM * oldw0_buf[j];
        w0_buf[j] += new_dw;
        oldw0_buf[j] = new_dw;
    }
    memcpy(w, w0_buf, sizeof(float)*16);
    memcpy(oldw, oldw0_buf, sizeof(float)*16);
}

}