#include"lc_dilate.h"


extern "C" {
float lc_dilate_stencil_core(float img_sample[STREL_ROWS * STREL_COLS])
{
    float max = 0.0;
    for (int i = 0; i < STREL_ROWS; i++)
#pragma HLS unroll
        for (int j = 0; j < STREL_COLS; j++) {
#pragma HLS unroll
            float temp = img_sample[i * STREL_COLS + j];
            if (temp > max) max = temp;
        }
    return max;
}
}


extern "C" {

void lc_dilate(float result[TILE_ROWS * GRID_COLS], float img[(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS], int which_boundary) 
{

    bool strel[25] = { 0, 0, 1, 0, 0, 
                       0, 1, 1, 1, 0, 
                       1, 1, 1, 1, 1, 
                       0, 1, 1, 1, 0, 
                       0, 0, 1, 0, 0 };

    int radius_p = STREL_ROWS / 2;
    int i;

    float img_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + PARA_FACTOR];
#pragma HLS array_partition variable=img_rf complete dim=0



    for (i = 0; i < GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR; i++) {
#pragma HLS pipeline II=1
#pragma HLS unroll
        img_rf[i + MAX_RADIUS] = img[i];
    }




    for (i = 0; i < GRID_COLS / PARA_FACTOR * TILE_ROWS; i++) {

        int k;
#pragma HLS pipeline II=1

        for (k = 0; k < PARA_FACTOR; k++) {
#pragma HLS unroll

            int m;
            float img_sample[STREL_ROWS * STREL_COLS];
#pragma HLS array_partition variable=img_sample complete dim=0

            for (m = 0; m < STREL_ROWS; m++) {
#pragma HLS unroll
                int n;
                for (n = 0; n < STREL_COLS; n++) {
#pragma HLS unroll
                    if ((i % (GRID_COLS / PARA_FACTOR) * PARA_FACTOR - MAX_RADIUS + n < 0) || (i % (GRID_COLS / PARA_FACTOR) * PARA_FACTOR - MAX_RADIUS + n >= GRID_COLS) || (which_boundary == TOP && (i < GRID_COLS / PARA_FACTOR) && m < MAX_RADIUS) || (which_boundary == BOTTOM && (i >= GRID_COLS / PARA_FACTOR * (TILE_ROWS - 1)) && m > MAX_RADIUS) || strel[m * STREL_COLS + n] != 1  ) img_sample[m * STREL_COLS + n] = 0;
                    else img_sample[m * STREL_COLS + n] = img_rf[GRID_COLS * m + n + k];

                }

            }

            result[i * PARA_FACTOR + k] = lc_dilate_stencil_core(img_sample);

        }


        for (k = 0; k < GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2; k++) {
#pragma HLS unroll
            img_rf[k] = img_rf[k + PARA_FACTOR];
        }

        for (k = 0; k < PARA_FACTOR; k++) {
#pragma HLS unroll
            img_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + k] = img[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + (i + 1) * PARA_FACTOR + k];
        }

    } 

    return;                                                             

}
}


extern "C" {

void buffer_load_img(int flag, int k, float img_dest[GRID_COLS * (TILE_ROWS + 2)], float *img_src)
{
#pragma HLS inline off
    if (flag) memcpy(img_dest, img_src + (k * TILE_ROWS * GRID_COLS - MAX_RADIUS * GRID_COLS) , sizeof(float) * ((unsigned long)((TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS)));
    return;
}
}

extern "C"{
void buffer_compute(int flag, float result_inner[GRID_COLS * TILE_ROWS], float img_inner[GRID_COLS * (TILE_ROWS + 2 * MAX_RADIUS)], int k)
{
#pragma HLS inline off
    if (flag) lc_dilate(result_inner, img_inner, k);
    return;
}
}

extern "C"{
void buffer_store(int flag, int k, float *result_dest, float result_src[GRID_COLS * TILE_ROWS])
{
#pragma HLS inline off
    if (flag) memcpy((result_dest + (k * TILE_ROWS * GRID_COLS)), result_src, sizeof(float) * ((unsigned long)(TILE_ROWS * GRID_COLS)));
    return;
}
}


extern "C" {

void workload(float *result, float *img)
{

    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=result
    #pragma HLS INTERFACE m_axi port=img offset=slave bundle=img
    
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=img bundle=control
    
    #pragma HLS INTERFACE s_axilite port=return bundle=control

        int i, n, k;

    float result_inner_0   [TILE_ROWS * GRID_COLS];
#pragma HLS array_partition variable=result_inner_0 cyclic factor=32
    float img_inner_0   [(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS];
#pragma HLS array_partition variable=img_inner_0 cyclic factor=32



    float result_inner_1   [TILE_ROWS * GRID_COLS];
#pragma HLS array_partition variable=result_inner_1 cyclic factor=32
    float img_inner_1   [(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS];
#pragma HLS array_partition variable=img_inner_1 cyclic factor=32



    float result_inner_2   [TILE_ROWS * GRID_COLS];
#pragma HLS array_partition variable=result_inner_2 cyclic factor=32
    float img_inner_2   [(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS];
#pragma HLS array_partition variable=img_inner_2 cyclic factor=32






for (k = 0; k < GRID_ROWS / TILE_ROWS + 2; k++) {
    int load_img_flag = k >= 0 && k < GRID_ROWS / TILE_ROWS;
    int compute_flag = k >= 1 && k < GRID_ROWS / TILE_ROWS + 1;
    int store_flag = k >= 2 && k < GRID_ROWS / TILE_ROWS + 2;
    
    if (k % 3 == 0) {
        buffer_load_img(load_img_flag, k, img_inner_0, img);
        buffer_compute(compute_flag, result_inner_2, img_inner_2, k - 1);
        buffer_store(store_flag, k - 2, result, result_inner_1);
    }

    else if (k % 3 == 1) {
        buffer_load_img(load_img_flag, k, img_inner_1, img);
        buffer_compute(compute_flag, result_inner_0, img_inner_0, k - 1);
        buffer_store(store_flag, k - 2, result, result_inner_2);
    }
    
    else {
        buffer_load_img(load_img_flag, k, img_inner_2, img);
        buffer_compute(compute_flag, result_inner_1, img_inner_1, k - 1);
        buffer_store(store_flag, k - 2, result, result_inner_0);
    }

}










return;

}

}


