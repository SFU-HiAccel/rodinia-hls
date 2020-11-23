#include <math.h>
#include <string.h>
#define ETA 0.3       //eta value
#define TILE_SIZE 8192
#define DUP_SIZE 32
extern "C" {


void buffer_load(int flag, float *conn, float conn_buf[TILE_SIZE][16]) {
#pragma HLS INLINE off
    if(flag) {
        memcpy(conn_buf[0], conn, sizeof(float) * TILE_SIZE * 16);
    }
}

void buffer_compute(int flag, int kk, float l1_buf[65537], float conn_buf[TILE_SIZE][16], float sum[16]) {
#pragma HLS INLINE off
    int j, k;
    if (flag) {
        LOOP2: for (k = 0; k < TILE_SIZE; k++) {
        #pragma HLS PIPELINE
            LOOP3: for (j = 0; j < 16; j++) {
            #pragma HLS UNROLL
                float product = conn_buf[k][j] * l1_buf[k + kk];
                sum[j] += product;
            }
        }
    }
}

void workload(float l1[65537], float l2[17], float conn[65537 * 16]) {
#pragma HLS INTERFACE m_axi port=l1 offset=slave bundle=l11
#pragma HLS INTERFACE m_axi port=l2 offset=slave bundle=l21
#pragma HLS INTERFACE m_axi port=conn offset=slave bundle=conn1


#pragma HLS INTERFACE s_axilite port=l1 bundle=control
#pragma HLS INTERFACE s_axilite port=l2 bundle=control
#pragma HLS INTERFACE s_axilite port=conn bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    //float sum;
    float sum[16];
    #pragma HLS ARRAY_PARTITION variable=sum complete dim=1

    int i, j, kk;

    float l1_buf[65536];

    float conn_buf_x[TILE_SIZE][16];
    #pragma HLS ARRAY_PARTITION variable=conn_buf_x complete dim=2

    float conn_buf_y[TILE_SIZE][16];
    #pragma HLS ARRAY_PARTITION variable=conn_buf_y complete dim=2


    float l2_buf[16];
    #pragma HLS ARRAY_PARTITION variable=l2_buf complete dim=1

    memcpy(l1_buf, l1+1, sizeof(float) * 65536);
    memcpy(sum, conn, sizeof(float) * 16);

    LOOP1: for (kk = 0; kk < 65536+TILE_SIZE; kk += TILE_SIZE) {
        int load_flag = ( kk < 65536 );
        int compute_flag = ( kk >= TILE_SIZE );
        i = (kk/TILE_SIZE)%2;

        if (i == 0) {
            buffer_load(load_flag, conn+(kk+1)*16, conn_buf_x);
            buffer_compute(compute_flag, kk, l1_buf, conn_buf_y, sum);
        }
        else {
            buffer_load(load_flag, conn+(kk+1)*16, conn_buf_y);
            buffer_compute(compute_flag, kk, l1_buf, conn_buf_x, sum);
        }
    }

    LOOP4: for (j = 0; j < 16; j++) {
    #pragma HLS UNROLL
        l2_buf[j] = (1.0 / (1.0 + exp(-sum[j])));
    }
    
    memcpy(l2+1, l2_buf, sizeof(float) * 16);
}
}
