#include "dilate.h"

extern "C" {

	#define PARA_FACTOR 32

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

	void lc_dilate(float result[TILE_ROWS * GRID_COLS], float img[(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS], int which_boundary)
	{

	    bool strel[25] = { 0, 0, 1, 0, 0,
	                       0, 1, 1, 1, 0,
	                       1, 1, 1, 1, 1,
	                       0, 1, 1, 1, 0,
	                       0, 0, 1, 0, 0 };

	    int radius_p = STREL_ROWS / 2;

	    float img_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + PARA_FACTOR];
//#pragma HLS array_partition variable=img_rf complete dim=0
#pragma HLS array_partition variable=img_rf block factor=1024 dim=0


	    for (int i = 0; i < MAX_RADIUS; i++) {
#pragma HLS pipeline II=1
#pragma HLS unroll
	        img_rf[i] = 0.0;
	    }

	    for (int i = 0; i < GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR; i++) {
#pragma HLS pipeline II=1
#pragma HLS unroll
	        img_rf[i + MAX_RADIUS] = img[i];
	    }

	    for (int i = 0; i < GRID_COLS / PARA_FACTOR * TILE_ROWS; i++) {
#pragma HLS pipeline II=1

	        for (int k = 0; k < PARA_FACTOR; k++) {
#pragma HLS unroll
	            float img_sample[STREL_ROWS * STREL_COLS];
#pragma HLS array_partition variable=img_sample complete dim=0

	            for (int m = 0; m < STREL_ROWS; m++) {
#pragma HLS unroll
	                for (int n = 0; n < STREL_COLS; n++) {
#pragma HLS unroll
	                    if (((i % (GRID_COLS / PARA_FACTOR)) * PARA_FACTOR - MAX_RADIUS + n + k < 0) ||
	                    	((i % (GRID_COLS / PARA_FACTOR)) * PARA_FACTOR - MAX_RADIUS + n + k >= GRID_COLS) ||
							(which_boundary == TOP && (i < GRID_COLS / PARA_FACTOR) && m < MAX_RADIUS) ||
							(which_boundary == BOTTOM && (i >= GRID_COLS / PARA_FACTOR * (TILE_ROWS - 1)) && m > MAX_RADIUS) ||
							strel[m * STREL_COLS + n] != 1  )
	                    {
	                    	img_sample[m * STREL_COLS + n] = 0;
	                    }
	                    else {
	                    	img_sample[m * STREL_COLS + n] = img_rf[GRID_COLS * m + n + k];
	                    }
	                }
	            }
	            result[i * PARA_FACTOR + k] = lc_dilate_stencil_core(img_sample);
	        }

	        for (int k = 0; k < GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2; k++) {
#pragma HLS unroll
	            img_rf[k] = img_rf[k + PARA_FACTOR];
	        }

	        for (int k = 0; k < PARA_FACTOR; k++) {
#pragma HLS unroll
	        	if ((GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + (i + 1) * PARA_FACTOR + k) < ((TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS))
	        		img_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + k] =
	        		   img[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + (i + 1) * PARA_FACTOR + k];
	        	else
	        		img_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + k] = 0.0;
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