#include <math.h>
#include <string.h>
#define ETA 0.3       //eta value
#define MOMENTUM 0.3  //momentum value

extern "C" {
void workload(float delta[17], float ly[65537], float w[65537 * 17], float oldw[65537 * 17]) {
#pragma HLS INTERFACE m_axi port=delta offset=slave bundle=delta1
#pragma HLS INTERFACE m_axi port=ly offset=slave bundle=ly1
#pragma HLS INTERFACE m_axi port=w offset=slave bundle=w1
#pragma HLS INTERFACE m_axi port=oldw offset=slave bundle=oldw1

#pragma HLS INTERFACE s_axilite port=delta bundle=control
#pragma HLS INTERFACE s_axilite port=ly bundle=control
#pragma HLS INTERFACE s_axilite port=w bundle=control
#pragma HLS INTERFACE s_axilite port=oldw bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    float new_dw;
    int k, kk, j;

    ly[0] = 1.0;

    for (j = 1; j <= 16; j++) {
        for (k = 0; k <= 65536; k++) {
            new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k * 17 + j]));
            w[k * 17 + j] += new_dw;
            oldw[k * 17 + j] = new_dw;
        }
    }
}

}
