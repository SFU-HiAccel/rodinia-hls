#include <math.h>
#include <string.h>
#define ETA 0.3       //eta value
#define TILE_SIZE 512

extern "C" {
void workload(float l1[65537], float l2[17], float conn[65537 * 16]) {
#pragma HLS INTERFACE m_axi port=l1 offset=slave bundle=l1
#pragma HLS INTERFACE m_axi port=l2 offset=slave bundle=l2
#pragma HLS INTERFACE m_axi port=conn offset=slave bundle=conn


#pragma HLS INTERFACE s_axilite port=l1 bundle=control
#pragma HLS INTERFACE s_axilite port=l2 bundle=control
#pragma HLS INTERFACE s_axilite port=conn bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    float sum;
    int j, k, kk;
    float l1_buf[65536];
    float conn_buf[TILE_SIZE*16];
    float l2_buf[16];
    float conn0_buf[16];

    memcpy(l1_buf, l1+1, sizeof(float) * 65536);
    memcpy(conn0_buf, conn, sizeof(float) * 16);

    /*** For each unit in second layer ***/
    for (j = 0; j < 16; j++) {

        /*** Compute weighted sum of its inputs ***/
        sum = conn0_buf[j];

        for (kk = 1; kk < 65537; kk += TILE_SIZE) {
            memcpy(conn_buf, conn+kk*16, sizeof(float) * TILE_SIZE * 16);
            for (k = 0; k < TILE_SIZE; k++) {
                float product = conn_buf[k * 16 + j] * l1_buf[k+kk-1];
                sum += product;
            }
        }

        l2_buf[j] = (1.0 / (1.0 + exp(-sum)));
    }
    memcpy(l2+1, l2_buf, sizeof(float) * 16);
}
}