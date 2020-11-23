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
#define SIZE_1 16
#include "../../../../common/mars_wide_bus_2d.h"
#undef SIZE_1

#define ETA 0.3       //eta value
#define MOMENTUM 0.3  //momentum value
#define TILE_SIZE 4096

extern "C" {
void buffer_load(int flag, int kk, class ap_uint< 512 > *w, float w_buf[TILE_SIZE][16], class ap_uint< 512 > *oldw, float oldw_buf[TILE_SIZE][16]) {
#pragma HLS INLINE off
    if(flag) {
        // #pragma HLS DATAFLOW
        memcpy_wide_bus_read_float_2d_16(w_buf, 0, 0, w, sizeof(float) * (kk+1)*16, sizeof(float) * TILE_SIZE * 16);
        memcpy_wide_bus_read_float_2d_16(oldw_buf, 0, 0, oldw, sizeof(float) * (kk+1)*16, sizeof(float) * TILE_SIZE * 16);
    }
}

void buffer_load_w(int flag, int kk, class ap_uint< 512 > *w, float w_buf[TILE_SIZE][16]){
    if(flag){
        memcpy_wide_bus_read_float_2d_16(w_buf, 0, 0, w, sizeof(float) * (kk+1)*16, sizeof(float) * TILE_SIZE * 16);
    }
}

void buffer_load_oldw(int flag, int kk, class ap_uint< 512 > *oldw, float oldw_buf[TILE_SIZE][16]){
    if(flag){
        memcpy_wide_bus_read_float_2d_16(oldw_buf, 0, 0, oldw, sizeof(float) * (kk+1)*16, sizeof(float) * TILE_SIZE * 16);
    }
}

void buffer_load_ly(int flag, int kk, class ap_uint< 512 > *ly, float ly_buf[TILE_SIZE]){
    if(flag){
        memcpy_wide_bus_read_float(ly_buf, ly, (kk+1)*sizeof(float), sizeof(float) * TILE_SIZE);
    }
}


void buffer_store(int flag, int kk, class ap_uint< 512 > *w, float w_buf[TILE_SIZE][16], class ap_uint< 512 > *oldw, float oldw_buf[TILE_SIZE][16]) {
#pragma HLS INLINE off
    if(flag) {
        memcpy_wide_bus_write_float_2d_16(w, w_buf, 0, 0, sizeof(float) * 16 * (kk+1-2*TILE_SIZE), sizeof(float) * TILE_SIZE * 16);
        memcpy_wide_bus_write_float_2d_16(oldw, oldw_buf, 0, 0, sizeof(float) * 16 * (kk+1-2*TILE_SIZE), sizeof(float) * TILE_SIZE * 16);
    }
}

void buffer_store_w(int flag, int kk, class ap_uint< 512 > *w, float w_buf[TILE_SIZE][16]){
    if(flag){
        memcpy_wide_bus_write_float_2d_16(w, w_buf, 0, 0, sizeof(float) * 16 * (kk+1-2*TILE_SIZE), sizeof(float) * TILE_SIZE * 16);
    }
}

void buffer_store_oldw(int flag, int kk, class ap_uint< 512 > *oldw, float oldw_buf[TILE_SIZE][16]){
    if(flag){
        memcpy_wide_bus_write_float_2d_16(oldw, oldw_buf, 0, 0, sizeof(float) * 16 * (kk+1-2*TILE_SIZE), sizeof(float) * TILE_SIZE * 16);
    }
}

void buffer_compute(int flag, int kk, float delta_buf[16], float ly_buf[TILE_SIZE], float w_buf[TILE_SIZE][16], float oldw_buf[TILE_SIZE][16]) {
#pragma HLS INLINE off
    float new_dw;
    int j, k;
    if (flag) {
        LOOP2:for (k = 0; k < TILE_SIZE; k++) {
        #pragma HLS PIPELINE
            LOOP3: for (j = 0; j < 16; j++) {
            #pragma HLS UNROLL
                new_dw = (float)ETA * (float)delta_buf[j] * (float)ly_buf[k] + (float)MOMENTUM * (float)oldw_buf[k][j];
                w_buf[k][j] += new_dw;
                oldw_buf[k][j] = new_dw;
            }
        }
    }
}

void workload(class ap_uint< 512 > *delta, class ap_uint< 512 > *ly, class ap_uint< 512 > *w, class ap_uint< 512 > *oldw) {
#pragma HLS INTERFACE m_axi port=delta offset=slave bundle=delta1
#pragma HLS INTERFACE m_axi port=ly offset=slave bundle=ly1
#pragma HLS INTERFACE m_axi port=w offset=slave bundle=w1
#pragma HLS INTERFACE m_axi port=oldw offset=slave bundle=oldw1

#pragma HLS INTERFACE s_axilite port=delta bundle=control
#pragma HLS INTERFACE s_axilite port=ly bundle=control
#pragma HLS INTERFACE s_axilite port=w bundle=control
#pragma HLS INTERFACE s_axilite port=oldw bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW

#pragma HLS stable variable=ly
#pragma HLS stable variable=delta

    float new_dw;
    int kk, i, j;

    float w0_buf[16];
#pragma HLS ARRAY_PARTITION variable=w0_buf complete dim=1
    float oldw0_buf[16];
#pragma HLS ARRAY_PARTITION variable=oldw0_buf complete dim=1
    float delta_buf[16];
#pragma HLS ARRAY_PARTITION variable=delta_buf complete dim=1

    //float ly_buf[65536];

    float ly_buf_x[TILE_SIZE];
    float ly_buf_y[TILE_SIZE];
    float ly_buf_z[TILE_SIZE];

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

	for(int loop = 0; loop < 1000; ++loop) {
        memcpy_wide_bus_read_float(delta_buf, delta, sizeof(float), sizeof(float) * 16);
        //memcpy_wide_bus_read_float(ly_buf, ly, sizeof(float), sizeof(float) * 65536);
        memcpy_wide_bus_read_float(w0_buf, w, 0, sizeof(float) * 16);
        memcpy_wide_bus_read_float(oldw0_buf, oldw, 0, sizeof(float) * 16);

        LOOP1:
        for (kk = 0; kk < 65536 + 2 * TILE_SIZE; kk += TILE_SIZE) {
            int load_flag = (kk < 65536);
            int compute_flag = ((kk >= TILE_SIZE) && (kk < 65536 + TILE_SIZE));
            int store_flag = (kk >= 2 * TILE_SIZE);
            i = (kk / TILE_SIZE) % 3;

            if (i == 0) {
                //buffer_load(load_flag, kk, w, w_buf_x, oldw, oldw_buf_x);
                buffer_load_w(load_flag, kk, w, w_buf_x);
                buffer_load_oldw(load_flag, kk, oldw, oldw_buf_x);
                buffer_load_ly(load_flag, kk, ly, ly_buf_x);
                buffer_compute(compute_flag, kk, delta_buf, ly_buf_z, w_buf_z, oldw_buf_z);
                //buffer_store(store_flag, kk, w, w_buf_y, oldw, oldw_buf_y);
                buffer_store_w(store_flag, kk, w, w_buf_y);
                buffer_store_oldw(store_flag, kk, oldw, oldw_buf_y);
            } else if (i == 1) {
                // buffer_load(load_flag, kk, w, w_buf_y, oldw, oldw_buf_y);
                buffer_load_w(load_flag, kk, w, w_buf_y);
                buffer_load_oldw(load_flag, kk, oldw, oldw_buf_y);
                buffer_load_ly(load_flag, kk, ly, ly_buf_y);
                buffer_compute(compute_flag, kk, delta_buf, ly_buf_x, w_buf_x, oldw_buf_x);
                //buffer_store(store_flag, kk, w, w_buf_z, oldw, oldw_buf_z);
                buffer_store_w(store_flag, kk, w, w_buf_z);
                buffer_store_oldw(store_flag, kk, oldw, oldw_buf_z);
            } else {
                // buffer_load(load_flag, kk, w, w_buf_z, oldw, oldw_buf_z);
                buffer_load_w(load_flag, kk, w, w_buf_z);
                buffer_load_oldw(load_flag, kk, oldw, oldw_buf_z);
                buffer_load_ly(load_flag, kk, ly, ly_buf_z);
                buffer_compute(compute_flag, kk, delta_buf, ly_buf_y, w_buf_y, oldw_buf_y);
                //buffer_store(store_flag, kk, w, w_buf_x, oldw, oldw_buf_x);
                buffer_store_w(store_flag, kk, w, w_buf_x);
                buffer_store_oldw(store_flag, kk , oldw, oldw_buf_x);
            }

        }

        LOOP4:
        for (j = 0; j < 16; j++) {
            #pragma HLS UNROLL
            new_dw = (float) ETA * (float) delta_buf[j] + (float) MOMENTUM * (float) oldw0_buf[j];
            w0_buf[j] += new_dw;
            oldw0_buf[j] = new_dw;
        }

        memcpy_wide_bus_write_float(w, w0_buf, 0, sizeof(float) * 16);
        memcpy_wide_bus_write_float(oldw, oldw0_buf, 0, sizeof(float) * 16);
    }
}

}
