#include "lud.h"
#include "../../../common/mc.h"
#define coalescing

#ifdef coalescing 
extern "C" {
void internal_load(int flag, float buffer[BSIZE][matrix_dim - BSIZE], class ap_uint<512> * src, int k, int idx){
	if(flag){
		for(int i = 0; i < BSIZE; i++){ //16 
			memcpy_wide_bus_read_float(buffer[i], src + ((idx + BSIZE) * GRID_COLS + idx + BSIZE + k * BSIZE * GRID_COLS + i * GRID_COLS) / (512 / 32), 0, sizeof(float) * (matrix_dim - BSIZE)); //sizeof(float) * (matrix_dim - idx - BSIZE));
			// memcpy(buffer[i], src + (idx + BSIZE) * GRID_COLS + idx + BSIZE + k * BSIZE * GRID_COLS + i * GRID_COLS, sizeof(float) * (matrix_dim - idx - BSIZE));
		}
	}
}

void internal_store(int flag, float buffer[BSIZE][matrix_dim - BSIZE], class ap_uint<512> * dst, int k, int idx){
	if(flag){
		for(int i = 0; i < BSIZE; i++){
			memcpy_wide_bus_write_float((class ap_uint<512> *)dst + ((idx + BSIZE) * GRID_COLS + idx + BSIZE + k * BSIZE * GRID_COLS + i * GRID_COLS) / (512 /32), buffer[i], 0, sizeof(float) * (matrix_dim - BSIZE));// sizeof(float) * (matrix_dim - idx - BSIZE));
			// memcpy(dst + (idx + BSIZE) * GRID_COLS + idx + BSIZE + k * BSIZE * GRID_COLS + i * GRID_COLS, buffer[i], sizeof(float) * (matrix_dim - idx - BSIZE));
		}
	}
}

#else

void internal_load(int flag, float buffer[BSIZE][matrix_dim - BSIZE], float * src, int k, int idx){
	if(flag){
		for(int i = 0; i < BSIZE; i++){ //16 
			memcpy(buffer[i], src + (idx + BSIZE) * GRID_COLS + idx + BSIZE + k * BSIZE * GRID_COLS + i * GRID_COLS, sizeof(float) * (matrix_dim - idx - BSIZE));
		}
	}
}

void internal_store(int flag, float buffer[BSIZE][matrix_dim - BSIZE], float * dst, int k, int idx){
	if(flag){
		for(int i = 0; i < BSIZE; i++){
			memcpy(dst + (idx + BSIZE) * GRID_COLS + idx + BSIZE + k * BSIZE * GRID_COLS + i * GRID_COLS, buffer[i], sizeof(float) * (matrix_dim - idx - BSIZE));
		}
	}
}

#endif

void internal_compute(int flag, float buffer_inner[BSIZE][matrix_dim - BSIZE], float buffer_left[GRID_ROWS][BSIZE], float buffer_up[BSIZE][GRID_COLS], int k, int idx){
	if(flag){
		
		for(int j = 0; j < matrix_dim - idx - BSIZE; j++){
			#pragma HLS pipeline II=1
			for(int i = 0; i < BSIZE; i++){
				#pragma HLS pipeline
				float tmp = buffer_inner[i][j];
				for(int m = 0; m < BSIZE; m++){
					#pragma HLS unroll
					tmp -= buffer_left[i + k * BSIZE][m] * 
							buffer_up[m][j + BSIZE];
				}
				buffer_inner[i][j] = tmp;
			}
		}
	}
}

void workload(class ap_uint<512> * result){
	#pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem		
	#pragma HLS INTERFACE s_axilite port=result bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	float buffer_up[BSIZE][GRID_COLS]; //16 * 256
	#pragma HLS array_partition variable=buffer_up complete dim=1
	float buffer_up_2[BSIZE][GRID_COLS];
	#pragma HLS array_partition variable=buffer_up_2 complete dim=2
	float buffer_left[GRID_ROWS][BSIZE];
	#pragma HLS array_partition variable=buffer_left complete dim=2
	float buffer_left_2[GRID_ROWS][BSIZE];
	#pragma HLS array_partition variable=buffer_left_2 complete dim=1

	outmost:for(int i = 0; i < matrix_dim; i += BSIZE){

		//load up and left
#ifdef coalescing
		l_dup:for(int j = 0; j < BSIZE; j++){
			memcpy_wide_bus_read_float(buffer_up[j], result + ((i + j) * GRID_COLS + i) / (512 / 32), 0,
				sizeof(float) * (GRID_COLS - i));
			memcpy_wide_bus_read_float(buffer_up_2[j], result + ((i + j) * GRID_COLS + i) / (512 / 32), 0,
				sizeof(float) * (GRID_COLS - i));
			// memcpy(buffer_up[j], result + (i + j) * GRID_COLS + i, 
			// 	sizeof(float) * (GRID_COLS - i));
		}

		l_dleft:for(int j = 0; j < GRID_ROWS - i - BSIZE; j++){
			memcpy_wide_bus_read_float(buffer_left[j], result + ((i + BSIZE) * GRID_COLS + j * GRID_COLS + i)/(512/32), 0,
				sizeof(float) * BSIZE); 
			memcpy_wide_bus_read_float(buffer_left_2[j], result + ((i + BSIZE) * GRID_COLS + j * GRID_COLS + i)/(512/32), 0,
				sizeof(float) * BSIZE);
			// memcpy(buffer_left[j], result + (i + BSIZE) * GRID_COLS + j * GRID_COLS + i,
			// 	sizeof(float) * BSIZE);
		}
#else
		l_dup:for(int j = 0; j < BSIZE; j++){
			memcpy(buffer_up[j], result + (i + j) * GRID_COLS + i, 
				sizeof(float) * (GRID_COLS - i));
			memcpy(buffer_up_2[j], result + (i + j) * GRID_COLS + i, 
				sizeof(float) * (GRID_COLS - i));
		}

		l_dleft:for(int j = 0; j < GRID_ROWS - i - BSIZE; j++){
			memcpy(buffer_left[j], result + (i + BSIZE) * GRID_COLS + j * GRID_COLS + i,
				sizeof(float) * BSIZE);
			memcpy(buffer_left_2[j], result + (i + BSIZE) * GRID_COLS + j * GRID_COLS + i,
				sizeof(float) * BSIZE);
		}
#endif
		//Process diagonal and store
		diagonal:for(int m = 0; m < BSIZE; m++){
			//#pragma HLS dependence
			p_dup:for(int n = m; n < BSIZE; n++){
				#pragma HLS pipeline
				float tmp = buffer_up[m][n];
				for(int k = 0; k < BSIZE; k++){                      //2. k<BSIZE  //1. pipeline]
					#pragma HLS unroll
					if(k < m)
						tmp -= buffer_up_2[m][k] * buffer_up[k][n];  //2. buffer_up_1 / 2
				}

				buffer_up[m][n] = tmp;
				buffer_up_2[m][n] = tmp;
			}

			float temp = 1.f / buffer_up[m][m];
			p_dleft:for(int n = m + 1; n < BSIZE; n++){
				#pragma HLS pipeline
				float tmp = buffer_up[n][m];
				for(int k = 0; k < BSIZE; k++){                      //pipeline
					#pragma HLS unroll
					if(k < m)
						tmp -= buffer_up_2[n][k] * buffer_up[k][m];
				}
				tmp = tmp * temp;
				buffer_up[n][m] = tmp;
				buffer_up_2[n][m] = tmp;

			}
		}

#ifdef coalescing
		if(i == matrix_dim - BSIZE){
			innermost:for(int j = 0; j < BSIZE; j++){
				memcpy_wide_bus_write_float(result + (i * GRID_COLS + j * GRID_COLS + i)/(512/32), buffer_up[j], 0,
					sizeof(float) * BSIZE);
				// memcpy(result + i * GRID_COLS + j * GRID_COLS + i, buffer_up[j],
					// sizeof(float) * BSIZE);
			}
		}
#else
		if(i == matrix_dim - BSIZE){
			innermost:for(int j = 0; j < BSIZE; j++){
				memcpy(result + i * GRID_COLS + j * GRID_COLS + i, buffer_up[j],
					sizeof(float) * BSIZE);
			}
		}
#endif

		//process perimeters
		int chunk_idx, chunk_num;

		chunk_num = (matrix_dim - (i + BSIZE)) / BSIZE;

		perimeters:for(chunk_idx = 0; chunk_idx < chunk_num; chunk_idx++){
			// p_up:for(int n = 0; n < BSIZE; n++){
			// 	for(int m = 0; m < BSIZE; m++){
			// 		for(int k = 0; k < m; k++){
			// 			buffer_up[m][n + chunk_idx * BSIZE + BSIZE] -= buffer_up[m][k] * buffer_up[k][n + chunk_idx * BSIZE + BSIZE];
			// 		}
			// 	}
			// }

			// Move m loop inward
			// p_left:for(int m = 0; m < BSIZE; m++){
			// 	for(int n = 0; n < BSIZE; n++){
			// 		float tmp = buffer_left[m + chunk_idx * BSIZE][n];
			// 		#pragma HLS pipeline
			// 		for(int k = 0; k < BSIZE; k++){
			// 			#pragma HLS unroll
			// 			if(k < n)
			// 				tmp -= buffer_left[m + chunk_idx * BSIZE][k] * buffer_up[k][n];
			// 			//tmp -= buffer_left[m + chunk_idx * BSIZE][k] * buffer_up[k][n];
			// 		}
			// 		float temp = buffer_up[n][n];
			// 		buffer_left[m + chunk_idx * BSIZE ][n] = tmp / temp;
			// 	}
			// }

			p_up:for(int m = 0; m < BSIZE; m++){

				for(int k = 0; k < m; k++){
					#pragma HLS pipeline

					float tmp = buffer_up[m][k];

					for(int n = 0; n < BSIZE; n++){
						#pragma HLS unroll
						buffer_up_2[m][n + chunk_idx * BSIZE + BSIZE] -= tmp * buffer_up_2[k][n + chunk_idx * BSIZE + BSIZE];
						//buffer_up[m][n + chunk_idx * BSIZE + BSIZE] = buffer_up_2[m][n + chunk_idx * BSIZE + BSIZE];
						//buffer_up[m][n + chunk_idx * BSIZE + BSIZE] -= tmp * buffer_up_2[k][n + chunk_idx * BSIZE + BSIZE];
					}
				}

				for(int n = 0; n < BSIZE; n++){
						#pragma HLS pipeline
						buffer_up[m][n + chunk_idx * BSIZE + BSIZE] = buffer_up_2[m][n + chunk_idx * BSIZE + BSIZE];
				}
			}

			p_left:for(int n = 0; n < BSIZE; n++){
				// float tmp = buffer_left[m + chunk_idx * BSIZE][n];
				// #pragma HLS pipeline
				//#pragma HLS dependence
				p_left_c:for(int k = 0; k < n; k++){
					#pragma HLS pipeline
					for(int m = 0; m < BSIZE; m++){
						#pragma HLS unroll
						buffer_left_2[m + chunk_idx * BSIZE][n] -= 
							buffer_left_2[m + chunk_idx * BSIZE][k] * buffer_up[k][n];
					}
					
				}

				p_left_s:for(int m = 0; m < BSIZE; m++){
					#pragma HLS unroll
					float temp = buffer_left_2[m + chunk_idx * BSIZE][n] / buffer_up[n][n];
					buffer_left[m + chunk_idx * BSIZE ][n] = temp;
					buffer_left_2[m + chunk_idx * BSIZE ][n] = temp;
				}
			}
		}

		//Store up and left
#ifdef coalescing
		s_up:for(int j = 0; j < BSIZE; j++){
			memcpy_wide_bus_write_float(result + (i * GRID_COLS + j * GRID_COLS + i)/(512/32), buffer_up[j], 0, sizeof(float) * (matrix_dim - i));
			// memcpy(result + (i + j) * GRID_COLS + i, buffer_up[j], sizeof(float) * (matrix_dim - i));
		}

		s_load:for(int j = 0; j < matrix_dim - i - BSIZE; j++){
			memcpy_wide_bus_write_float(result + ((i + BSIZE + j) * GRID_COLS + i) /(512/32), buffer_left[j], 0, sizeof(float) * BSIZE);
			// memcpy(result + (i + BSIZE + j) * GRID_COLS + i, buffer_left[j], sizeof(float) * BSIZE); 
		}
#else
		s_up:for(int j = 0; j < BSIZE; j++){
			memcpy(result + (i + j) * GRID_COLS + i, buffer_up[j], sizeof(float) * (matrix_dim - i));
		}

		s_load:for(int j = 0; j < matrix_dim - i - BSIZE; j++){
			memcpy(result + (i + BSIZE + j) * GRID_COLS + i, buffer_left[j], sizeof(float) * BSIZE); 
		}
#endif
		//Process internal part
		float buffer_inner_0[BSIZE][matrix_dim - BSIZE];
		#pragma HLS array_partition variable=buffer_inner_0 complete dim=1
		float buffer_inner_1[BSIZE][matrix_dim - BSIZE];
		#pragma HLS array_partition variable=buffer_inner_1 complete dim=1
		float buffer_inner_2[BSIZE][matrix_dim - BSIZE];
		#pragma HLS array_partition variable=buffer_inner_2 complete dim=1

		internal:for(int k = 0; k < chunk_num + 2; k++){
			int load_flag = k >=0 && k < chunk_num;
			int compute_flag = k >= 1 && k < chunk_num + 1;
			int store_flag = k >= 2 && k < chunk_num + 2;

			if(k % 3 == 0){
				internal_load(load_flag, buffer_inner_0, result, k, i);
				internal_compute(compute_flag, buffer_inner_2, buffer_left, buffer_up, k - 1, i);
				internal_store(store_flag, buffer_inner_1, result, k - 2, i);
			} else if(k % 3 == 1){
				internal_load(load_flag, buffer_inner_1, result, k, i);
				internal_compute(compute_flag, buffer_inner_0, buffer_left, buffer_up, k - 1, i);
				internal_store(store_flag, buffer_inner_2, result, k - 2, i);
			} else {
				internal_load(load_flag, buffer_inner_2, result, k, i);
				internal_compute(compute_flag, buffer_inner_1, buffer_left, buffer_up, k - 1, i);
				internal_store(store_flag, buffer_inner_0, result, k - 2, i);
			}
		} 

	}

	return;
}
}
