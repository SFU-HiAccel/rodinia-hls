#include"lud.h"

#include "../../../common/mc.h"

extern "C" {

void buffer_load(int load_flag, int kkk, float mat_reg_inner[TILE_ROWS * (SIZE - PARA_FACTOR)],class ap_uint<LARGE_BUS> * result, int kk)
{
#pragma HLS inline off
	if (load_flag) {
		for (int i = 0; i < TILE_ROWS; i++) {
			memcpy_wide_bus_read_float(mat_reg_inner + i * (SIZE - PARA_FACTOR), result + ((kk + 1) * PARA_FACTOR * SIZE + (kk + 1) * PARA_FACTOR + kkk * SIZE * TILE_ROWS + i * SIZE) / (LARGE_BUS / 32), 0, sizeof(float) * (SIZE - (0 + 1) * PARA_FACTOR));
		
		}
	}

	return;
}


void buffer_compute(int compute_flag, int kkk, float mat_ram_left[SIZE - PARA_FACTOR][PARA_FACTOR], float mat_reg_inner[TILE_ROWS * (SIZE - PARA_FACTOR)], float mat_result_inner[TILE_ROWS * (SIZE - PARA_FACTOR)], float mat_ram_up[PARA_FACTOR][SIZE], int kk)
{

#pragma HLS inline off
	if (compute_flag) {
		for (int j = 0; j < SIZE - (0 + 1) * TILE_ROWS; j++) {
#pragma HLS pipeline II=1
			for (int i = 0; i < TILE_ROWS; i++) {
#pragma HLS unroll
				float sum = 0;
				float temp = mat_reg_inner[i * (SIZE - PARA_FACTOR) + j];
				for (int k = 0; k < PARA_FACTOR; k++) {
#pragma HLS unroll
					sum += mat_ram_left[i + TILE_ROWS * kkk][k] * mat_ram_up[k][j + PARA_FACTOR];
				}
				mat_result_inner[i * (SIZE - PARA_FACTOR) + j] = temp - sum;
			}
		}
	}



}


void buffer_store(int store_flag, int kkk, float mat_reg_inner[TILE_ROWS * (SIZE - PARA_FACTOR)], class ap_uint<LARGE_BUS> * result, int kk)
{
#pragma HLS inline off
	if (store_flag) {
		for (int i = 0; i < TILE_ROWS; i++) {
			memcpy_wide_bus_write_float(result + ((kk + 1) * TILE_ROWS * SIZE + (kk + 1) * TILE_ROWS + kkk * SIZE * TILE_ROWS + i * SIZE) / (LARGE_BUS / 32), mat_reg_inner + i * (SIZE - TILE_ROWS), 0, sizeof(float) * (SIZE - (0 + 1) * TILE_ROWS));
		}
	}
}








void workload(class ap_uint<LARGE_BUS> * result)

{

    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem
    
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    
    #pragma HLS INTERFACE s_axilite port=return bundle=control


	int i, j, k;



	int kk;
L_kk:	for (kk = 0; kk < SIZE / PARA_FACTOR; kk++) {
		float mat_ram_up[PARA_FACTOR][SIZE];
#pragma HLS array_partition variable=mat_ram_up complete dim=1
		float mat_ram_left[SIZE - PARA_FACTOR][PARA_FACTOR];
#pragma HLS array_partition variable=mat_ram_left complete dim=0

		// load up
L_loadup:	for (i = 0; i < PARA_FACTOR; i++) {
#pragma HLS pipeline II=1
			memcpy_wide_bus_read_float(mat_ram_up[i], result + (kk * PARA_FACTOR * SIZE + i * SIZE + kk * PARA_FACTOR) / (LARGE_BUS / 32), 0, sizeof(float) * (SIZE - kk * PARA_FACTOR));
		}

		// load left
L_loadleft:	for (i = 0; i < SIZE - (kk + 1) * PARA_FACTOR; i++) {
#pragma HLS pipeline II=1
			memcpy_wide_bus_read_float(mat_ram_left[i], result + ((kk + 1) * PARA_FACTOR * SIZE + i * SIZE + kk * PARA_FACTOR) / (LARGE_BUS / 32), 0, sizeof(float) * PARA_FACTOR);
		}

		// calculate left up
L_calcleftup:	for (i = 0; i < PARA_FACTOR; i++) {
#pragma HLS unroll
			for (j = i; j < PARA_FACTOR; j++) {
#pragma HLS unroll
				for (k = 0; k < i; k++) {
#pragma HLS unroll
					mat_ram_up[i][j] -= mat_ram_up[i][k] * mat_ram_up[k][j];
				}
			}

			float temp = 1.f / mat_ram_up[i][i];
			for (j = i + 1; j < PARA_FACTOR; j++) {
#pragma HLS unroll
				for (k = 0; k < i; k++) {
#pragma HLS unroll
					mat_ram_up[j][i] -= mat_ram_up[j][k] * mat_ram_up[k][i];
				}
				mat_ram_up[j][i] *= temp;
			}
		}

		if (kk == SIZE / PARA_FACTOR - 1) {
			for (i = 0; i < PARA_FACTOR; i++) {
#pragma HLS pipeline II=1
				memcpy_wide_bus_write_float(result + (kk * PARA_FACTOR * SIZE + i * SIZE + kk * PARA_FACTOR) / (LARGE_BUS / 32), mat_ram_up[i], 0, sizeof(float) * (SIZE - kk * PARA_FACTOR));
			}
		}

		// calculate perimeters

		int chunk_idx, size_inter, chunks_in_inter_row;

		size_inter = SIZE - (kk + 1) * PARA_FACTOR;
		chunks_in_inter_row = size_inter / PARA_FACTOR;


L_perimeters:	for (chunk_idx = 0; chunk_idx < chunks_in_inter_row; chunk_idx++) {

			// do calculate of up tile
L_calcup:		for (j = 0; j < PARA_FACTOR; j++) {
#pragma HLS pipeline II = 1
				for (i = 0; i < PARA_FACTOR; i++) {
#pragma HLS unroll
					for (k = 0; k < i; k++) {
#pragma HLS unroll
						mat_ram_up[i][j + chunk_idx * PARA_FACTOR + PARA_FACTOR] -= mat_ram_up[i][k] * mat_ram_up[k][j + chunk_idx * PARA_FACTOR + PARA_FACTOR];
					}
				}
			}

			// do calculate of left tile
L_calcleft:		for (i = 0; i < PARA_FACTOR; i++) {
#pragma HLS pipeline II = 1
				for (j = 0; j < PARA_FACTOR; j++) {
#pragma HLS unroll
					for (k = 0; k < j; k++) {
#pragma HLS unroll
						mat_ram_left[i + chunk_idx * PARA_FACTOR][j] -= mat_ram_left[i + chunk_idx * PARA_FACTOR][k] * mat_ram_up[k][j];
					}
					float temp = mat_ram_up[j][j];
					mat_ram_left[i + chunk_idx * PARA_FACTOR][j] /= temp;
				}
			}

		}
			// store back the up and left tile

L_storeup:		for (i = 0; i < PARA_FACTOR; i++) {
#pragma HLS pipeline II = 1
				memcpy_wide_bus_write_float(result + (kk * PARA_FACTOR * SIZE + i * SIZE + kk * PARA_FACTOR) / (LARGE_BUS / 32), mat_ram_up[i], 0, sizeof(float) * (SIZE - kk * PARA_FACTOR));
			}

L_storeleft:		for (i = 0; i < SIZE - (kk + 1) * PARA_FACTOR; i++) {
#pragma HLS pipeline II = 1
				memcpy_wide_bus_write_float(result + ((kk + 1) * PARA_FACTOR * SIZE + i * SIZE + kk * PARA_FACTOR) / (LARGE_BUS / 32), mat_ram_left[i], 0, sizeof(float) * PARA_FACTOR);
			}



			int kkk;
			float mat_reg_inner_0[TILE_ROWS * (SIZE - PARA_FACTOR)];
#pragma HLS array_partition variable=mat_reg_inner_0 cyclic factor=8 dim=0
			float mat_reg_inner_1[TILE_ROWS * (SIZE - PARA_FACTOR)];
#pragma HLS array_partition variable=mat_reg_inner_1 cyclic factor=8 dim=0
			float mat_reg_inner_2[TILE_ROWS * (SIZE - PARA_FACTOR)];
#pragma HLS array_partition variable=mat_reg_inner_2 cyclic factor=8 dim=0

			float mat_result_inner_0[TILE_ROWS * (SIZE - PARA_FACTOR)];
#pragma HLS array_partition variable=mat_reg_inner_0 cyclic factor=8 dim=0
			float mat_result_inner_1[TILE_ROWS * (SIZE - PARA_FACTOR)];
#pragma HLS array_partition variable=mat_reg_inner_1 cyclic factor=8 dim=0
			float mat_result_inner_2[TILE_ROWS * (SIZE - PARA_FACTOR)];
#pragma HLS array_partition variable=mat_reg_inner_2 cyclic factor=8 dim=0




chunks_in_inter_row = size_inter / TILE_ROWS;

L_kkk:			for (kkk = 0; kkk < chunks_in_inter_row + 2; kkk++) {
				int load_flag = kkk >= 0 && kkk < chunks_in_inter_row;
				int compute_flag = kkk >= 1 && kkk < chunks_in_inter_row + 1;
				int store_flag = kkk >= 2 && kkk < chunks_in_inter_row + 2;

				if (kkk % 3 == 0) {
					buffer_load(load_flag, kkk, mat_reg_inner_0, result, kk);
					buffer_compute(compute_flag, kkk - 1, mat_ram_left, mat_reg_inner_2, mat_result_inner_2, mat_ram_up, kk);
					buffer_store(store_flag, kkk - 2, mat_result_inner_1, result, kk);
				}

				else if (kkk % 3 == 1) {
					buffer_load(load_flag, kkk, mat_reg_inner_1, result, kk);
					buffer_compute(compute_flag, kkk - 1, mat_ram_left, mat_reg_inner_0, mat_result_inner_0, mat_ram_up, kk);
					buffer_store(store_flag, kkk - 2, mat_result_inner_2, result, kk);
				}

				else {
					buffer_load(load_flag, kkk, mat_reg_inner_2, result, kk);
					buffer_compute(compute_flag, kkk - 1, mat_ram_left, mat_reg_inner_1, mat_result_inner_1, mat_ram_up, kk);
					buffer_store(store_flag, kkk - 2, mat_result_inner_0, result, kk);
				}

			}

	}



    return;

}








}

