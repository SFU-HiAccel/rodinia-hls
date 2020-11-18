#include"lc_gicov.h"
// #define PARA_FACTOR 1

float lc_gicov_stencil_core(float grad_x_sample[NPOINTS], float grad_y_sample[NPOINTS], float sin_angle[NPOINTS], float cos_angle[NPOINTS])
{
    float Grad[NPOINTS];
    int i;
    for (i = 0; i < NPOINTS; i++) {
        Grad[i] = grad_x_sample[i] * cos_angle[i] + grad_y_sample[i] * sin_angle[i];
    }
    int n;
    float sum = 0.0;
    for (n = 0; n < NPOINTS; n++) sum += Grad[n];
    float mean = sum / (float)NPOINTS;

    // Compute the variance of the gradient values
    float var = 0.0;
    for (n = 0; n < NPOINTS; n++) {
        sum = Grad[n] - mean;
        var += sum * sum;
    }
    var = var / (float)(NPOINTS - 1);

    // Keep track of the maximal GICOV value seen so far
    return mean * mean / var;
}

extern "C" {

void lc_gicov(float result[TILE_ROWS * GRID_COLS], float grad_x[(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS], float grad_y[(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS], int which_boundary) 
{
    float sin_angle[16] = {0, 0.382683432365090, 0.707106781186548, 0.923879532511287, 1, 0.923879532511287, 0.707106781186548, 0.382683432365090, 1.22464679914735e-16, -0.382683432365090, -0.707106781186548, -0.923879532511287, -1, -0.923879532511287, -0.707106781186548, -0.382683432365090};
    float cos_angle[16] = {1, 0.923879532511287, 0.707106781186548, 0.382683432365090, 6.12323399573677e-17, -0.382683432365090, -0.707106781186548, -0.923879532511287, -1, -0.923879532511287, -0.707106781186548, -0.382683432365090, -1.83697019872103e-16, 0.382683432365090, 0.707106781186547, 0.923879532511287};

    int tX[2][16] = {{1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0}, {2, 1, 1, 0, 0, -1, -2, -2, -2, -2, -2, -1, 0, 0, 1, 1}};
    int tY[2][16] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1}, {0, 0, 1, 1, 1, 1, 1, 0, 0, -1, -2, -2, -2, -2, -2, -1}};


    int i;

    float grad_x_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + PARA_FACTOR];
    #pragma HLS array_partition variable=grad_x_rf complete dim=0
    float grad_y_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + PARA_FACTOR];
    #pragma HLS array_partition variable=grad_y_rf complete dim=0

    for (i = 0; i < GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR; i++) {
        #pragma HLS pipeline II=1
        #pragma HLS unroll
        grad_x_rf[i + MAX_RADIUS] = grad_x[i]; 
        grad_y_rf[i + MAX_RADIUS] = grad_y[i];
    }

    for (i = 0; i < GRID_COLS / PARA_FACTOR * TILE_ROWS; i++) {
        int k;
        #pragma HLS pipeline II=1
        for (k = 0; k < PARA_FACTOR; k++) {
            #pragma HLS unroll
            if ((which_boundary == TOP && i < GRID_COLS / PARA_FACTOR * MAX_RADIUS) || (which_boundary == BOTTOM && i >= GRID_COLS / PARA_FACTOR * (TILE_ROWS - MAX_RADIUS)) || (PARA_FACTOR * (i % (GRID_COLS / PARA_FACTOR)) + k < MAX_RADIUS) || (GRID_COLS - (PARA_FACTOR * (i % (GRID_COLS / PARA_FACTOR)) + k) <= MAX_RADIUS)) {
                result[i * PARA_FACTOR + k] = 0;
                continue;
            }

            int t;
            float gicov_max = 0;
            for (t = 0; t < NCIRCLES; t++) {
                #pragma HLS unroll
                float grad_x_sample[NPOINTS];
                float grad_y_sample[NPOINTS];

                int j;
                
                for (j = 0; j < NPOINTS; j++) {
                    #pragma HLS unroll
                    int x, y;
                    y = MAX_RADIUS + tY[t][j];
                    x = MAX_RADIUS + tX[t][j];
                    grad_x_sample[j] = grad_x_rf[y * GRID_COLS + x + k];
                    grad_y_sample[j] = grad_y_rf[y * GRID_COLS + x + k];
                }

                float gicov_local = lc_gicov_stencil_core(grad_x_sample, grad_y_sample, sin_angle, cos_angle);
                
                if (gicov_local > gicov_max) {
                    result[i * PARA_FACTOR + k] = sqrt(gicov_local);
                    gicov_max = gicov_local;
                }
            }
        }

        for (k = 0; k < GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2; k++) {
            #pragma HLS unroll //Change factor of unrolling 
            grad_x_rf[k] = grad_x_rf[k + PARA_FACTOR];
            grad_y_rf[k] = grad_y_rf[k + PARA_FACTOR];
        }

        for (k = 0; k < PARA_FACTOR; k++) {
            #pragma HLS unroll
            grad_x_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + k] = grad_x[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + (i + 1) * PARA_FACTOR + k];
            grad_y_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + k] = grad_y[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + (i + 1) * PARA_FACTOR + k];
        }
    }                                                              
}
}


extern "C" {

void workload(float result[GRID_ROWS * GRID_COLS], float grad_x[GRID_ROWS * GRID_COLS], float grad_y[GRID_ROWS * GRID_COLS])
{

    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=result
    #pragma HLS INTERFACE m_axi port=grad_x offset=slave bundle=grad_x
    #pragma HLS INTERFACE m_axi port=grad_y offset=slave bundle=grad_y
    
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=grad_x bundle=control
    #pragma HLS INTERFACE s_axilite port=grad_y bundle=control
    
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    int i, n, k;

    float result_inner   [TILE_ROWS * GRID_COLS];
    float grad_x_inner   [(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS];
    float grad_y_inner   [(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS];


    for (k = 0; k < GRID_ROWS / TILE_ROWS; k++) {
        memcpy(grad_x_inner, grad_x + k * TILE_ROWS * GRID_COLS - MAX_RADIUS * GRID_COLS, sizeof(float) * (TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS);
        memcpy(grad_y_inner, grad_y + k * TILE_ROWS * GRID_COLS - MAX_RADIUS * GRID_COLS, sizeof(float) * (TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS);

        lc_gicov(result_inner, grad_x_inner, grad_y_inner, k);
        
        memcpy(result + k * TILE_ROWS * GRID_COLS, result_inner, sizeof(float) * (TILE_ROWS * GRID_COLS));
    }
    return;
}

}


