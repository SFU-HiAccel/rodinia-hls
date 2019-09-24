#include"lud.h"


extern "C" {



void buffer_load(int load_flag, int kkk, float mat_reg_inner[TILE_ROWS][SIZE - TILE_ROWS], float* result, int kk)
{
#pragma HLS inline off
	if (load_flag) {
		for (int i = 0; i < TILE_ROWS; i++) {
			memcpy(mat_reg_inner[i], result + (kk + 1) * TILE_ROWS * SIZE + (kk + 1) * TILE_ROWS + kkk * SIZE * TILE_ROWS + i * SIZE, sizeof(float) * (SIZE - (0 + 1) * TILE_ROWS));
		}
	}

	return;
}


void buffer_compute(int compute_flag, int kkk, float mat_ram_left[SIZE - TILE_ROWS][TILE_ROWS], float mat_reg_inner[TILE_ROWS][SIZE - TILE_ROWS], float mat_ram_up[TILE_ROWS][SIZE], int kk)
{
#pragma HLS inline off
	if (compute_flag) {
		for (int i = 0; i < TILE_ROWS; i++) {
			for (int j = 0; j < SIZE - (kk + 1) * TILE_ROWS; j++) {
				for (int k = 0; k < TILE_ROWS; k++) {
					mat_reg_inner[i][j] -= mat_ram_left[i + TILE_ROWS * kkk][k] * mat_ram_up[k][j + TILE_ROWS];
				}
			}
		}
	}
}


void buffer_store(int store_flag, int kkk, float mat_reg_inner[TILE_ROWS][SIZE - TILE_ROWS], float* result, int kk)
{
#pragma HLS inline off
	if (store_flag) {
		for (int i = 0; i < TILE_ROWS; i++) {
			memcpy(result + (kk + 1) * TILE_ROWS * SIZE + (kk + 1) * TILE_ROWS + kkk * SIZE * TILE_ROWS + i * SIZE, mat_reg_inner[i], sizeof(float) * (SIZE - (0 + 1) * TILE_ROWS));
		}
	}
}








void workload(float result[GRID_ROWS * GRID_COLS])
{

    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=result
    
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    
    #pragma HLS INTERFACE s_axilite port=return bundle=control


	int i, j, k;



	int kk;
L_kk:	for (kk = 0; kk < SIZE / TILE_ROWS; kk++) {
		float mat_ram_up[TILE_ROWS][SIZE];
#pragma HLS array_partition variable=mat_ram_up complete dim=1
		float mat_ram_left[SIZE - TILE_ROWS][TILE_ROWS];
#pragma HLS array_partition variable=mat_ram_left complete dim=2

		// load up
L_loadup:	for (i = 0; i < TILE_ROWS; i++) {
#pragma HLS pipeline II=1
			memcpy(mat_ram_up[i], result + kk * TILE_ROWS * SIZE + i * SIZE + kk * TILE_ROWS, sizeof(float) * (SIZE - kk * TILE_ROWS));
		}

		// load left
L_loadleft:	for (i = 0; i < SIZE - (kk + 1) * TILE_ROWS; i++) {
#pragma HLS pipeline II=1
			memcpy(mat_ram_left[i], result + (kk + 1) * TILE_ROWS * SIZE + i * SIZE + kk * TILE_ROWS, sizeof(float) * TILE_ROWS);
		}

		// calculate left up
L_calcleftup:	for (i = 0; i < TILE_ROWS; i++) {
#pragma HLS unroll
			for (j = i; j < TILE_ROWS; j++) {
#pragma HLS unroll
				for (k = 0; k < i; k++) {
#pragma HLS unroll
					mat_ram_up[i][j] -= mat_ram_up[i][k] * mat_ram_up[k][j];
				}
			}

			float temp = 1.f / mat_ram_up[i][i];
			for (j = i + 1; j < TILE_ROWS; j++) {
#pragma HLS unroll
				for (k = 0; k < i; k++) {
#pragma HLS unroll
					mat_ram_up[j][i] -= mat_ram_up[j][k] * mat_ram_up[k][i];
				}
				mat_ram_up[j][i] *= temp;
			}
		}

		if (kk == SIZE / TILE_ROWS - 1) {
L_store_lastupleft:	for (i = 0; i < TILE_ROWS; i++) {
#pragma HLS pipeline II=1
				memcpy(result + kk * TILE_ROWS * SIZE + i * SIZE + kk * TILE_ROWS, mat_ram_up[i], sizeof(float) * (SIZE - kk * TILE_ROWS));
			}
		}

		// calculate perimeters

		int chunk_idx, size_inter, chunks_in_inter_row;

		size_inter = SIZE - (kk + 1) * TILE_ROWS;
		chunks_in_inter_row = size_inter / TILE_ROWS;


L_perimeters:	for (chunk_idx = 0; chunk_idx < chunks_in_inter_row; chunk_idx++) {

			// do calculate of up tile
L_calcup:		for (j = 0; j < TILE_ROWS; j++) {
#pragma HLS pipeline II = 1
				for (i = 0; i < TILE_ROWS; i++) {
#pragma HLS unroll
					for (k = 0; k < i; k++) {
#pragma HLS unroll
						mat_ram_up[i][j + chunk_idx * TILE_ROWS + TILE_ROWS] -= mat_ram_up[i][k] * mat_ram_up[k][j + chunk_idx * TILE_ROWS + TILE_ROWS];
					}
				}
			}

			// do calculate of left tile
L_calcleft:		for (i = 0; i < TILE_ROWS; i++) {
#pragma HLS pipeline II = 1
				for (j = 0; j < TILE_ROWS; j++) {
#pragma HLS unroll
					for (k = 0; k < j; k++) {
#pragma HLS unroll
						mat_ram_left[i + chunk_idx * TILE_ROWS][j] -= mat_ram_left[i + chunk_idx * TILE_ROWS][k] * mat_ram_up[k][j];
					}
					float temp = mat_ram_up[j][j];
					mat_ram_left[i + chunk_idx * TILE_ROWS][j] /= temp;
				}
			}

		}
			// store back the up and left tile

L_storeup:		for (i = 0; i < TILE_ROWS; i++) {
#pragma HLS pipeline II=1
				memcpy(result + kk * TILE_ROWS * SIZE + i * SIZE + kk * TILE_ROWS, mat_ram_up[i], sizeof(float) * (SIZE - kk * TILE_ROWS));
			}

L_storeleft:		for (i = 0; i < SIZE - (kk + 1) * TILE_ROWS; i++) {
#pragma HLS pipeline II=1
				memcpy(result + (kk + 1) * TILE_ROWS * SIZE + i * SIZE + kk * TILE_ROWS, mat_ram_left[i], sizeof(float) * TILE_ROWS);
			}



			int kkk;


			float mat_reg_inner[TILE_ROWS][SIZE - TILE_ROWS];
L_kkk:			for (kkk = 0; kkk < chunks_in_inter_row; kkk++) {

				buffer_load(1, kkk, mat_reg_inner, result, kk);
				buffer_compute(1, kkk, mat_ram_left, mat_reg_inner, mat_ram_up, kk);
				buffer_store(1, kkk, mat_reg_inner, result, kk);


			}

	

	}



    return;

}








}

