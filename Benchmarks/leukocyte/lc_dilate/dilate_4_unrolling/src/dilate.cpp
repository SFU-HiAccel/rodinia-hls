#include "dilate.h"

extern "C" {

	#define PARA_FACTOR 8

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

	void lc_dilate(float result[TILE_ROWS * (TILE_COLS+2*MAX_RADIUS)],
			float img [(TILE_ROWS + 2 * MAX_RADIUS) * (TILE_COLS + 2 * MAX_RADIUS)], int which_boundary)
	{

	    bool strel[25] = { 0, 0, 1, 0, 0,
	                       0, 1, 1, 1, 0,
	                       1, 1, 1, 1, 1,
	                       0, 1, 1, 1, 0,
	                       0, 0, 1, 0, 0 };

	    int radius_p = STREL_ROWS / 2;

	    float img_rf[(TILE_COLS+2*MAX_RADIUS) * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + PARA_FACTOR];
		#pragma HLS array_partition variable=img_rf complete dim=0

	    LOAD_WORKING_IMG_SET_BLANK : for (int i = 0; i < MAX_RADIUS; i++) {
#pragma HLS pipeline II=1
#pragma HLS unroll
	        img_rf[i] = 0.0;
	    }

	    LOAD_WORKING_IMG_SET : for (int i = 0; i < (TILE_COLS+2*MAX_RADIUS) * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR; i++) {
#pragma HLS pipeline II=1
#pragma HLS unroll
	        img_rf[i + MAX_RADIUS] = img[i];
	    }

	    COMPUTE_EACH_OUTPUT : for (int i = 0; i < ((TILE_COLS+2*MAX_RADIUS) * TILE_ROWS) / PARA_FACTOR ; i++) {
#pragma HLS pipeline II=1

	        UNROLL_PE : for (int k = 0; k < PARA_FACTOR; k++) {
#pragma HLS unroll
	            float img_sample[STREL_ROWS * STREL_COLS];
#pragma HLS array_partition variable=img_sample complete dim=0

	            FILTER_ROW : for (int m = 0; m < STREL_ROWS; m++) {
#pragma HLS unroll
	            	FILTER_COL : for (int n = 0; n < STREL_COLS; n++) {
#pragma HLS unroll
//	                    if ( ( ( i % ((TILE_COLS+2*MAX_RADIUS) / PARA_FACTOR) ) * PARA_FACTOR - MAX_RADIUS + n + k < 0 ) ||
//	                    	 ( (i % ((TILE_COLS+2*MAX_RADIUS) / PARA_FACTOR)) * PARA_FACTOR - MAX_RADIUS + n + k >= GRID_COLS ) ||
//							 (which_boundary == TOP && (i < (TILE_COLS+2*MAX_RADIUS) / PARA_FACTOR) && m < MAX_RADIUS) ||
//							 (which_boundary == BOTTOM && (i >= (TILE_COLS+2*MAX_RADIUS) / PARA_FACTOR * (TILE_ROWS - 1)) && m > MAX_RADIUS) ||
//	            		if ( ( ((i * PARA_FACTOR) % (TILE_COLS+2*MAX_RADIUS)) - MAX_RADIUS + n + k < 0) ||
//							 ( ((i * PARA_FACTOR) % (TILE_COLS+2*MAX_RADIUS)) - MAX_RADIUS + n + k >= GRID_COLS) ||
//							 (which_boundary == TOP && ((i * PARA_FACTOR) < (TILE_COLS+2*MAX_RADIUS)) && m < MAX_RADIUS) ||
//							 (which_boundary == BOTTOM && ((i * PARA_FACTOR) >= (TILE_COLS+2*MAX_RADIUS) * (TILE_ROWS - 1)) && m > MAX_RADIUS) ||
						if ( strel[m * STREL_COLS + n] != 1 )
	                    {
	                    	img_sample[m * STREL_COLS + n] = 0;
	                    }
	                    else {
	                    	img_sample[m * STREL_COLS + n] = img_rf[(TILE_COLS+2*MAX_RADIUS) * m + n + k];
	                    }
	                }
	            }
	            result[i * PARA_FACTOR + k] = lc_dilate_stencil_core(img_sample);
	        }

	        SHIFT_AHEAD_BODY_INDEX : for (int k = 0; k < (TILE_COLS+2*MAX_RADIUS) * (2 * MAX_RADIUS) + MAX_RADIUS * 2; k++) {
#pragma HLS unroll
	            img_rf[k] = img_rf[k + PARA_FACTOR];
	        }

	        SHIFT_AHEAD_LAST_INDEX : for (int k = 0; k < PARA_FACTOR; k++) {
#pragma HLS unroll
	        	if ((TILE_COLS+2*MAX_RADIUS) * (2 * MAX_RADIUS) + MAX_RADIUS + (i + 1) * PARA_FACTOR + k <
	        		(TILE_ROWS + 2 * MAX_RADIUS) * (TILE_COLS + 2 * MAX_RADIUS)){
					img_rf[(TILE_COLS+2*MAX_RADIUS) * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + k] =
							img[(TILE_COLS+2*MAX_RADIUS) * (2 * MAX_RADIUS) + MAX_RADIUS + (i + 1) * PARA_FACTOR + k];
	        	}else{
	        		img_rf[(TILE_COLS+2*MAX_RADIUS) * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + k] = 0.0;
	        	}
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

	    float result_inner_0   [TILE_ROWS * GRID_COLS];
	    float img_inner_0      [(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS];

	    float result_inner_col   [TILE_ROWS * (TILE_COLS+2*MAX_RADIUS)];
	    float img_inner_col      [(TILE_ROWS + 2 * MAX_RADIUS) * (TILE_COLS + 2 * MAX_RADIUS)];

	    int start_col=0;

		ROW_TILES : for (int k = 0; k < GRID_ROWS / TILE_ROWS; k++) {
			load_data_tile(img_inner_0, img, k);

			COL_TILES : for (int j = 0; j < GRID_COLS / TILE_COLS; ++j){

				if (j == 0){
					start_col = 0;
					LOAD_IMG_ROW_LEFTMOST : for (int row=0; row<(TILE_ROWS + 2 * MAX_RADIUS); ++row){
						LOAD_IMG_COL_LEFTMOST : for (int col=0; col<(TILE_COLS + MAX_RADIUS); ++col){
#pragma HLS PIPELINE II=1
							img_inner_col[row*(TILE_COLS + 2 * MAX_RADIUS)+MAX_RADIUS+col] = img_inner_0[row*GRID_COLS+col];
						}
					}
					LOAD_IMG_ROW_LEFTMOST_BLANK : for (int row=0; row<(TILE_ROWS + 2 * MAX_RADIUS); ++row){
						LOAD_IMG_COL_LEFTMOST_BLANK : for (int col=0; col<MAX_RADIUS; ++col){
#pragma HLS PIPELINE II=1
							img_inner_col[row*(TILE_COLS + 2 * MAX_RADIUS)+col] = 0.0;
						}
					}
				}
				else if (j == GRID_COLS / TILE_COLS-1){
					start_col = j * TILE_COLS - MAX_RADIUS;
					LOAD_IMG_ROW_RIGHTMOST : for (int row=0; row<(TILE_ROWS + 2 * MAX_RADIUS); ++row){
						LOAD_IMG_COL_RIGHTMOST : for (int col=0; col<(TILE_COLS + MAX_RADIUS); ++col){
#pragma HLS PIPELINE II=1
							img_inner_col[row*(TILE_COLS + 2 * MAX_RADIUS)+col] = img_inner_0[row*GRID_COLS+start_col+col];
						}
					}
					LOAD_IMG_ROW_RIGHTMOST_BLANK : for (int row=0; row<(TILE_ROWS + 2 * MAX_RADIUS); ++row){
						LOAD_IMG_COL_RIGHTMOST_BLANK : for (int col=0; col<MAX_RADIUS; ++col){
#pragma HLS PIPELINE II=1
							img_inner_col[row*(TILE_COLS + 2 * MAX_RADIUS)+(TILE_COLS + MAX_RADIUS)+col] = 0.0;
						}
					}
				}
				else{
					start_col = j * TILE_COLS - MAX_RADIUS;
					LOAD_IMG_ROW_REST : for (int row=0; row<(TILE_ROWS + 2 * MAX_RADIUS); ++row){
						LOAD_IMG_COL_REST : for (int col=0; col<(TILE_COLS + 2 * MAX_RADIUS); ++col){
#pragma HLS PIPELINE II=1
							img_inner_col[row*(TILE_COLS + 2 * MAX_RADIUS)+col] = img_inner_0[row*GRID_COLS+start_col+col];
						}
					}
				}

				lc_dilate(result_inner_col, img_inner_col, k);

				start_col = j * TILE_COLS;
				STORE_RST_ROW : for (int row=0; row<TILE_ROWS; ++row){
					STORE_RST_COL : for (int col=0; col<TILE_COLS; ++col){
#pragma HLS PIPELINE II=1
						result_inner_0[row*GRID_COLS+start_col+col] = result_inner_col[row*(TILE_COLS+2*MAX_RADIUS)+MAX_RADIUS+col];
					}
				}
			}

			store_result_tile(result_inner_0, result, k);
		}

		return;
	}

}