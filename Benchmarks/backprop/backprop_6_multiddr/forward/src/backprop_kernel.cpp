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
#define TILE_SIZE 8192
#define UNROLL_FACTOR 16
extern "C" {


void buffer_load(int flag, int kk, class ap_uint< 512 > *conn, float conn_buf[TILE_SIZE][16]) {
#pragma HLS INLINE off
    if(flag) {
        memcpy_wide_bus_read_float_2d_16(conn_buf, 0, 0, conn, sizeof(float) * (kk+1)*16, sizeof(float) * TILE_SIZE * 16);

    }
}

void buffer_compute(int flag, int kk, float l1_buf[65537], float conn_buf[TILE_SIZE][16], float sum[UNROLL_FACTOR][16]) {
#pragma HLS INLINE off
    int i, j, k;
    if (flag) {
        LOOP2: for (k = 0; k < TILE_SIZE; k+=UNROLL_FACTOR) {
        #pragma HLS PIPELINE
            for(i = 0; i < UNROLL_FACTOR; i++) {
            #pragma HLS UNROLL
                LOOP3: for (j = 0; j < 16; j++) {
                #pragma HLS UNROLL
                    float product = conn_buf[k+i][j] * l1_buf[k+i + kk];
                    sum[i][j] += product;
                }
            }
        }
    }
}

void workload(class ap_uint< 512 > *l1, class ap_uint< 512 > *l2, class ap_uint< 512 > *conn) {
#pragma HLS INTERFACE m_axi port=l1 offset=slave bundle=l11
#pragma HLS INTERFACE m_axi port=l2 offset=slave bundle=l21
#pragma HLS INTERFACE m_axi port=conn offset=slave bundle=conn1


#pragma HLS INTERFACE s_axilite port=l1 bundle=control
#pragma HLS INTERFACE s_axilite port=l2 bundle=control
#pragma HLS INTERFACE s_axilite port=conn bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

            float sum[UNROLL_FACTOR][16];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=0

            int i, j, kk;

            float l1_buf[65536];
#pragma HLS ARRAY_PARTITION variable=l1_buf cyclic factor=8 dim=1

            float conn_buf_x[TILE_SIZE][16];
#pragma HLS ARRAY_PARTITION variable=conn_buf_x cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=conn_buf_x complete dim=2

            float conn_buf_y[TILE_SIZE][16];
#pragma HLS ARRAY_PARTITION variable=conn_buf_y cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=conn_buf_y complete dim=2


            float l2_buf[16];
#pragma HLS ARRAY_PARTITION variable=l2_buf complete dim=1
			for(int loop = 0; loop < 2000; ++loop) {
            memcpy_wide_bus_read_float(l1_buf, l1, sizeof(float), sizeof(float) * 65536);

            memcpy_wide_bus_read_float(sum[0], conn, 0, sizeof(float) * 16);

            for (i = 1; i < UNROLL_FACTOR; i++) {
#pragma HLS UNROLL
                for (j = 0; j < 16; j++) {
#pragma HLS UNROLL
                    sum[i][j] = 0.0;
                }
            }

            LOOP1:
            for (kk = 0; kk < 65536 + TILE_SIZE; kk += TILE_SIZE) {
                int load_flag = (kk < 65536);
                int compute_flag = (kk >= TILE_SIZE);
                i = (kk / TILE_SIZE) % 2;

                if (i == 0) {
                    buffer_load(load_flag, kk, conn, conn_buf_x);
                    buffer_compute(compute_flag, kk, l1_buf, conn_buf_y, sum);
                } else {
                    buffer_load(load_flag, kk, conn, conn_buf_y);
                    buffer_compute(compute_flag, kk, l1_buf, conn_buf_x, sum);
                }
            }

            LOOP4:
            for (j = 0; j < 16; j++) {
#pragma HLS UNROLL
                l2_buf[j] = (float) (1.0 / (float) (1.0 +
                                                    (float) expf((float) (-(float) sum[0][j] - (float) sum[1][j] -
                                                                          (float) sum[2][j] - (float) sum[3][j] -
                                                                          (float) sum[4][j] - (float) sum[5][j] -
                                                                          (float) sum[6][j] - (float) sum[7][j] -
                                                                          (float) sum[8][j] - (float) sum[9][j] -
                                                                          (float) sum[10][j] - (float) sum[11][j] -
                                                                          (float) sum[12][j] - (float) sum[13][j] -
                                                                          (float) sum[14][j] - (float) sum[15][j]
                                                    ))));
            }

            memcpy_wide_bus_write_float(l2, l2_buf, sizeof(float), sizeof(float) * 16);
}
}
}
