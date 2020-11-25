#include "dilate.h"

extern "C" {

	const int PARA_FACTOR=16;

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

	void lc_dilate(int flag, float result[TILE_ROWS * (TILE_COLS+2*MAX_RADIUS)],
			float img [(TILE_ROWS + 2 * MAX_RADIUS) * (TILE_COLS + 2 * MAX_RADIUS)], int which_boundary)
	{
		if (flag){
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
		}
	    return;
	}

	void load_col_tile (int flag, int col_tile_idx,
			float col_tile [(TILE_ROWS + 2 * MAX_RADIUS) * (TILE_COLS + 2 * MAX_RADIUS)],
			float row_tile [(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS])
	{
		if (flag){
		int start_col = 0;
		if (col_tile_idx == 0){
			start_col = 0;
			LOAD_IMG_ROW_LEFTMOST : for (int row=0; row<(TILE_ROWS + 2 * MAX_RADIUS); ++row){
#pragma HLS PIPELINE II=1
				LOAD_IMG_COL_LEFTMOST : for (int col=0; col<(TILE_COLS + MAX_RADIUS); ++col){
#pragma HLS UNROLL
					col_tile[row*(TILE_COLS + 2 * MAX_RADIUS)+MAX_RADIUS+col] = row_tile[row*GRID_COLS+col];
				}
			}
			LOAD_IMG_ROW_LEFTMOST_BLANK : for (int row=0; row<(TILE_ROWS + 2 * MAX_RADIUS); ++row){
#pragma HLS PIPELINE II=1
				LOAD_IMG_COL_LEFTMOST_BLANK : for (int col=0; col<MAX_RADIUS; ++col){
#pragma HLS UNROLL
					col_tile[row*(TILE_COLS + 2 * MAX_RADIUS)+col] = 0.0;
				}
			}
		}
		else if (col_tile_idx == GRID_COLS / TILE_COLS-1){
			start_col = col_tile_idx * TILE_COLS - MAX_RADIUS;
			LOAD_IMG_ROW_RIGHTMOST : for (int row=0; row<(TILE_ROWS + 2 * MAX_RADIUS); ++row){
#pragma HLS PIPELINE II=1
				LOAD_IMG_COL_RIGHTMOST : for (int col=0; col<(TILE_COLS + MAX_RADIUS); ++col){
#pragma HLS UNROLL
					col_tile[row*(TILE_COLS + 2 * MAX_RADIUS)+col] = row_tile[row*GRID_COLS+start_col+col];
				}
			}
			LOAD_IMG_ROW_RIGHTMOST_BLANK : for (int row=0; row<(TILE_ROWS + 2 * MAX_RADIUS); ++row){
#pragma HLS PIPELINE II=1
				LOAD_IMG_COL_RIGHTMOST_BLANK : for (int col=0; col<MAX_RADIUS; ++col){
#pragma HLS UNROLL
					col_tile[row*(TILE_COLS + 2 * MAX_RADIUS)+(TILE_COLS + MAX_RADIUS)+col] = 0.0;
				}
			}
		}
		else{
			start_col = col_tile_idx * TILE_COLS - MAX_RADIUS;
			LOAD_IMG_ROW_REST : for (int row=0; row<(TILE_ROWS + 2 * MAX_RADIUS); ++row){
#pragma HLS PIPELINE II=1
				LOAD_IMG_COL_REST : for (int col=0; col<(TILE_COLS + 2 * MAX_RADIUS); ++col){
#pragma HLS UNROLL
					col_tile[row*(TILE_COLS + 2 * MAX_RADIUS)+col] = row_tile[row*GRID_COLS+start_col+col];
				}
			}
		}
		}
	}

	void store_col_tile (int flag, int col_tile_idx,
			float row_tile_result [TILE_ROWS * GRID_COLS],
			float col_tile_result [TILE_ROWS * (TILE_COLS+2*MAX_RADIUS)])
	{
		if (flag){
		int start_col = col_tile_idx * TILE_COLS;
		STORE_RST_ROW : for (int row=0; row<TILE_ROWS; ++row){
#pragma HLS PIPELINE II=1
			STORE_RST_COL : for (int col=0; col<TILE_COLS; ++col){
#pragma HLS UNROLL
				row_tile_result[row*GRID_COLS+start_col+col] = col_tile_result[row*(TILE_COLS+2*MAX_RADIUS)+MAX_RADIUS+col];
			}
		}
		}
	}

	void load_row_tile(int flag, float img_bram[(TILE_ROWS+2*MAX_RADIUS)*GRID_COLS], ap_uint<LARGE_BUS> *img, int tile_index)
	{
		if (flag){
		int starting_index = tile_index * TILE_ROWS * GRID_COLS / 16;
// 		for (int row=0; row<TILE_ROWS+2*MAX_RADIUS; ++row){
// 			for (int col=0; col<GRID_COLS / 16; ++col){
// #pragma HLS PIPELINE II=1
// 				memcpy_wide_bus_read_float(img_bram+(row*GRID_COLS+col*16), (class ap_uint<LARGE_BUS> *)(img+(starting_index+row*GRID_COLS/16+col)), 0, sizeof(float) * 16);
// 			}
// 		}
		memcpy_wide_bus_read_float(img_bram, (class ap_uint<LARGE_BUS> *)(img+starting_index), 0, sizeof(float) *((unsigned long)(TILE_ROWS+2*MAX_RADIUS)* GRID_COLS ));
		}
	}

	void compute_row_tile(int flag, int row_tile_idx,
			float row_tile_img [(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS],
			float row_tile_result [TILE_ROWS * GRID_COLS])
	{
		if (flag){
	    float col_tile_img_0 [(TILE_ROWS + 2 * MAX_RADIUS) * (TILE_COLS + 2 * MAX_RADIUS)];
#pragma HLS array_partition variable=col_tile_img_0 cyclic factor=PARA_FACTOR
		float col_tile_result_0 [TILE_ROWS * (TILE_COLS+2*MAX_RADIUS)];
#pragma HLS array_partition variable=col_tile_result_0 cyclic factor=PARA_FACTOR

	    float col_tile_img_1 [(TILE_ROWS + 2 * MAX_RADIUS) * (TILE_COLS + 2 * MAX_RADIUS)];
#pragma HLS array_partition variable=col_tile_img_1 cyclic factor=PARA_FACTOR
		float col_tile_result_1 [TILE_ROWS * (TILE_COLS+2*MAX_RADIUS)];
#pragma HLS array_partition variable=col_tile_result_1 cyclic factor=PARA_FACTOR

	    float col_tile_img_2 [(TILE_ROWS + 2 * MAX_RADIUS) * (TILE_COLS + 2 * MAX_RADIUS)];
#pragma HLS array_partition variable=col_tile_img_2 cyclic factor=PARA_FACTOR
		float col_tile_result_2 [TILE_ROWS * (TILE_COLS+2*MAX_RADIUS)];
#pragma HLS array_partition variable=col_tile_result_2 cyclic factor=PARA_FACTOR

		int NUM_COL_TILES = GRID_COLS / TILE_COLS;

		COL_TILES : for (int j = 0; j < NUM_COL_TILES + 2; ++j)
		{
		    int load_img_flag = j >= 0 && j < NUM_COL_TILES;
		    int compute_flag = j >= 1 && j < NUM_COL_TILES + 1;
		    int store_flag = j >= 2 && j < NUM_COL_TILES + 2;

		    if (j % 3 == 0){
				load_col_tile(load_img_flag, j, col_tile_img_0, row_tile_img);
				lc_dilate(compute_flag, col_tile_result_2, col_tile_img_2, row_tile_idx);
				store_col_tile(store_flag, j-2, row_tile_result, col_tile_result_1);
		    }
		    else if (j % 3 == 1){
				load_col_tile(load_img_flag, j, col_tile_img_1, row_tile_img);
				lc_dilate(compute_flag, col_tile_result_0, col_tile_img_0, row_tile_idx);
				store_col_tile(store_flag, j-2, row_tile_result, col_tile_result_2);
		    }
		    else{
				load_col_tile(load_img_flag, j, col_tile_img_2, row_tile_img);
				lc_dilate(compute_flag, col_tile_result_1, col_tile_img_1, row_tile_idx);
				store_col_tile(store_flag, j-2, row_tile_result, col_tile_result_0);
		    }
		}
		}
	}

	void store_row_tile(int flag, float result_bram[TILE_ROWS * GRID_COLS], ap_uint<LARGE_BUS>* result, int tile_index)
	{
		if (flag){
		int starting_index = tile_index * TILE_ROWS * GRID_COLS / 16;
// 		for (int row=0; row<TILE_ROWS; ++row){
// 			for (int col=0; col<GRID_COLS / 16; ++col){
// #pragma HLS PIPELINE II=1
// 				//memcpy_wide_bus_write_float((class ap_uint<LARGE_BUS> *)(result+(starting_index+row*GRID_COLS/16+col)), result_bram+(row*GRID_COLS+col*16), 0, 64);
// 				memcpy_wide_bus_write_float((class ap_uint<LARGE_BUS> *)(result+(starting_index+row*GRID_COLS/16+col)), result_bram+(row*GRID_COLS+col*16), 0, sizeof(float) * 16);
// 			}
// 			}

		memcpy_wide_bus_write_float((class ap_uint<LARGE_BUS> *)(result+starting_index), result_bram, 0, sizeof(float) * ((unsigned long) TILE_ROWS * GRID_COLS));
		}
	}

	void workload(ap_uint<LARGE_BUS> *result, ap_uint<LARGE_BUS>* img)
	{

	    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=result1
	    #pragma HLS INTERFACE m_axi port=img offset=slave bundle=img1

	    #pragma HLS INTERFACE s_axilite port=result bundle=control
	    #pragma HLS INTERFACE s_axilite port=img bundle=control

	    #pragma HLS INTERFACE s_axilite port=return bundle=control

		float row_tile_img_0 [(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS];
#pragma HLS array_partition variable=row_tile_img_0 cyclic factor=PARA_FACTOR
	    float row_tile_result_0 [TILE_ROWS * GRID_COLS];
#pragma HLS array_partition variable=row_tile_result_0 cyclic factor=PARA_FACTOR

		float row_tile_img_1 [(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS];
#pragma HLS array_partition variable=row_tile_img_1 cyclic factor=PARA_FACTOR
	    float row_tile_result_1 [TILE_ROWS * GRID_COLS];
#pragma HLS array_partition variable=row_tile_result_1 cyclic factor=PARA_FACTOR

		float row_tile_img_2 [(TILE_ROWS + 2 * MAX_RADIUS) * GRID_COLS];
#pragma HLS array_partition variable=row_tile_img_2 cyclic factor=PARA_FACTOR
	    float row_tile_result_2 [TILE_ROWS * GRID_COLS];
#pragma HLS array_partition variable=row_tile_result_2 cyclic factor=PARA_FACTOR

	    int NUM_ROW_TILES = GRID_ROWS / TILE_ROWS;

		ROW_TILES : for (int k = 0; k < NUM_ROW_TILES + 2; k++) {
		    int load_img_flag = k >= 0 && k < NUM_ROW_TILES;
		    int compute_flag = k >= 1 && k < NUM_ROW_TILES + 1;
		    int store_flag = k >= 2 && k < NUM_ROW_TILES + 2;

		    if (k % 3 == 0){
				load_row_tile(load_img_flag, row_tile_img_0, img, k);
				compute_row_tile(compute_flag, k-1, row_tile_img_2, row_tile_result_2);
				store_row_tile(store_flag, row_tile_result_1, result, k-2);
		    }
		    else if (k % 3 == 1){
				load_row_tile(load_img_flag, row_tile_img_1, img, k);
				compute_row_tile(compute_flag, k-1, row_tile_img_0, row_tile_result_0);
				store_row_tile(store_flag, row_tile_result_2, result, k-2);
		    }
		    else{
				load_row_tile(load_img_flag, row_tile_img_2, img, k);
				compute_row_tile(compute_flag, k-1, row_tile_img_1, row_tile_result_1);
				store_row_tile(store_flag, row_tile_result_0, result, k-2);
		    }
		}

		return;
	}

}
