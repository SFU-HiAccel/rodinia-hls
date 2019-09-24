#include"lc_dilate.h"


extern "C" {

void lc_dilate(float result[TILE_ROWS * GRID_COLS], float img[(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS], int which_boundary)
{






    

    bool strel[25] = { 0, 0, 1, 0, 0, 
                       0, 1, 1, 1, 0, 
                       1, 1, 1, 1, 1, 
                       0, 1, 1, 1, 0, 
                       0, 0, 1, 0, 0 };

    int radius_p = STREL_ROWS / 2;
    int radius_q = STREL_COLS / 2;

    for (int i = 0; i < TILE_ROWS; i++) {
        for (int j = 0; j < GRID_COLS; j++) {
            float max = 0.0, temp;
            for (int m = 0; m < STREL_ROWS; m++) {
                for (int n = 0; n < STREL_COLS; n++) {
                    int p = i - radius_p + m;
                    int q = j - radius_q + n;
                    if ((p >= 0 || which_boundary != TOP) && q >= 0 && (p < TILE_ROWS || which_boundary != BOTTOM) && q < GRID_COLS && strel[m * STREL_COLS + n] != 0) {

                        temp = img[p * GRID_COLS + q + GRID_COLS * MAX_RADIUS];
                        if (temp > max) max = temp;
                    }
                }
            }
            result[i * GRID_COLS + j] = max;
        }
    }


    return;
}

}





extern "C" {

void workload(float result[GRID_COLS * GRID_ROWS], float img[GRID_COLS * GRID_ROWS])
{

    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=result
    #pragma HLS INTERFACE m_axi port=img offset=slave bundle=img
    
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=img bundle=control
    
    #pragma HLS INTERFACE s_axilite port=return bundle=control

        int i, n, k;

    float result_inner_0   [TILE_ROWS * GRID_COLS];
    float img_inner_0      [(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS];

for (k = 0; k < GRID_ROWS / TILE_ROWS; k++) {


memcpy(img_inner_0, img + k * GRID_COLS * TILE_ROWS - MAX_RADIUS * GRID_COLS, sizeof(float) * (GRID_COLS * TILE_ROWS + 2 * MAX_RADIUS * GRID_COLS));




lc_dilate(result_inner_0, img_inner_0, k);

memcpy(result + k * GRID_COLS * TILE_ROWS, result_inner_0, sizeof(float) * GRID_COLS * TILE_ROWS);


}








return;

}

}


