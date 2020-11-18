#include "dilate.h"

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
	                    if ((p >= 0 || which_boundary != TOP) &&
	                    		q >= 0 &&
								(p < TILE_ROWS || which_boundary != BOTTOM) &&
								q < GRID_COLS &&
								strel[m * STREL_COLS + n] != 0)
	                    {
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

	void load_data_tile(float img_bram[(TILE_ROWS+2*MAX_RADIUS)*GRID_COLS], float img[(GRID_ROWS+2*MAX_RADIUS)*GRID_COLS], int tile_index)
	{
		int starting_index = tile_index * TILE_ROWS * GRID_COLS;
		for (int row=0; row<TILE_ROWS+2*MAX_RADIUS; ++row){
			for (int col=0; col<GRID_COLS; ++col){
#pragma HLS PIPELINE II=1
				img_bram[row*GRID_COLS+col] = img[starting_index+row*GRID_COLS+col];
			}
		}
	}

	void store_result_tile(float result_bram[TILE_ROWS * GRID_COLS], float result[GRID_ROWS*GRID_COLS], int tile_index)
	{
		int starting_index = tile_index * TILE_ROWS * GRID_COLS;
		for (int row=0; row<TILE_ROWS; ++row){
			for (int col=0; col<GRID_COLS; ++col){
#pragma HLS PIPELINE II=1
				result[starting_index+row*GRID_COLS+col] = result_bram[row*GRID_COLS+col];
			}
		}
	}

	void workload(float result[GRID_COLS * GRID_ROWS], float img[(GRID_ROWS+2*MAX_RADIUS)*GRID_COLS])
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
			load_data_tile(img_inner_0, img, k);
			lc_dilate(result_inner_0, img_inner_0, k);
			store_result_tile(result_inner_0, result, k);
		}

		return;
	}

}