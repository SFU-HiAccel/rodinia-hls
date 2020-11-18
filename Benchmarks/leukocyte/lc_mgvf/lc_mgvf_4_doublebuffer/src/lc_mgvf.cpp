#include "lc_mgvf.h"

#define __kernel
#define __global

float heaviside(float x) {
    // A simpler, faster approximation of the Heaviside function
    float out = 0.0;
    if (x > -0.0001) out = 0.5;
    if (x >  0.0001) out = 1.0;
    return out; 
}

float lc_mgvf_stencil_core(float c, float ul, float u, float ur, float l, float r, float dl, float d, float dr, float vI)
{
    float UL = ul - c;
    float U  = u  - c;
    float UR = ur - c;
    float L  = l  - c;
    float R  = r  - c;
    float DL = dl - c;
    float D  = d  - c;
    float DR = dr - c;

    float vHe = c + MU_O_LAMBDA * (
                                   heaviside(UL) * UL + 
                                   heaviside(U)  * U  + 
                                   heaviside(UR) * UR + 
                                   heaviside(L)  * L  + 
                                   heaviside(R)  * R  + 
                                   heaviside(DL) * DL + 
                                   heaviside(D)  * D  + 
                                   heaviside(DR) * DR 
                                   );

    float new_val = vHe - (ONE_O_LAMBDA * vI * (vHe - vI));


    return new_val;
}

void lc_mgvf(float result[TILE_ROWS * GRID_COLS], float imgvf[(TILE_ROWS + 2) * GRID_COLS], float I[TILE_ROWS * GRID_COLS], int which_boundary)
{
	int cols = GRID_COLS;
	int rows = GRID_ROWS;
	float imgvf_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + PARA_FACTOR];
#pragma HLS array_partition variable=imgvf_rf complete dim=0

	int i;
	for (i = 0; i < GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR; i++) {
#pragma HLS pipeline II=1
#pragma HLS unroll
		imgvf_rf[i + MAX_RADIUS] = imgvf[i];
	}

	for (i = 0; i < GRID_COLS / PARA_FACTOR * TILE_ROWS; i++) {
		int k;
#pragma HLS pipeline II=1

		for (k = 0; k < PARA_FACTOR; k++) {
#pragma HLS unroll
			float ul[PARA_FACTOR], u[PARA_FACTOR], ur[PARA_FACTOR], l[PARA_FACTOR], c[PARA_FACTOR], r[PARA_FACTOR], dl[PARA_FACTOR], d[PARA_FACTOR], dr[PARA_FACTOR], vI[PARA_FACTOR];

			int is_top = (which_boundary == TOP) && (i < GRID_COLS / PARA_FACTOR);
			int is_right = (i % (GRID_COLS / PARA_FACTOR) == (GRID_COLS / PARA_FACTOR - 1)) && (k == PARA_FACTOR - 1);
			int is_bottom = (which_boundary == BOTTOM) && (i >= GRID_COLS / PARA_FACTOR * (TILE_ROWS - 1));
			int is_left = (i % (GRID_COLS / PARA_FACTOR) == 0) && (k == 0);

			c[k] = imgvf_rf[GRID_COLS * (0) + 0 + k + GRID_COLS + MAX_RADIUS];
			ul[k] = (is_top || is_left) ? c[k] : imgvf_rf[GRID_COLS * (-1) + -1 + k + GRID_COLS + MAX_RADIUS];
			u[k] = (is_top) ? c[k] : imgvf_rf[GRID_COLS * (-1) + 0 + k + GRID_COLS + MAX_RADIUS];
			ur[k] = (is_top || is_right) ? c[k] : imgvf_rf[GRID_COLS * (-1) + 1 + k + GRID_COLS + MAX_RADIUS];
			l[k] = (is_left) ? c[k] : imgvf_rf[GRID_COLS * (0) + -1 + k + GRID_COLS + MAX_RADIUS];
			r[k] = (is_right) ? c[k] : imgvf_rf[GRID_COLS * (0) + 1 + k + GRID_COLS + MAX_RADIUS];
			dl[k] = (is_bottom || is_left) ? c[k] : imgvf_rf[GRID_COLS * (1) + -1 + k + GRID_COLS + MAX_RADIUS];
			d[k] = (is_bottom) ? c[k] : imgvf_rf[GRID_COLS * (1) + 0 + k + GRID_COLS + MAX_RADIUS];
			dr[k] = (is_bottom || is_right) ? c[k] : imgvf_rf[GRID_COLS * (1) + 1 + k + GRID_COLS + MAX_RADIUS];
			vI[k] = I[i * PARA_FACTOR + k];
			result[i * PARA_FACTOR + k] = lc_mgvf_stencil_core(c[k], ul[k], u[k], ur[k], l[k], r[k], dl[k], d[k], dr[k], vI[k]);
		}

		for (k = 0; k < GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2; k++) {
#pragma HLS unroll
			imgvf_rf[k] = imgvf_rf[k + PARA_FACTOR];
		}

		for (k = 0; k < PARA_FACTOR; k++) {
#pragma HLS unroll
			imgvf_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + k] = imgvf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + (i + 1) * PARA_FACTOR + k];
		}
	}

	return;
}


extern "C"{
void buffer_load_imgvf(int flag, int k, float imgvf_dest[GRID_COLS * (TILE_ROWS + 2)], float *imgvf_src)
{
#pragma HLS inline off
    if (flag) memcpy(imgvf_dest, imgvf_src + k * TILE_ROWS * GRID_COLS - GRID_COLS , sizeof(float) * (TILE_ROWS + 2) * GRID_COLS);
    return;
}
}

extern "C"{
void buffer_load_I(int flag, int k, float I_dest[GRID_COLS * TILE_ROWS], float *I_src)
{
#pragma HLS inline off
    if (flag) memcpy(I_dest, I_src + k * TILE_ROWS * GRID_COLS, sizeof(float) * TILE_ROWS * GRID_COLS);
    return;
}
}

extern "C"{
void buffer_compute(int flag, float result_inner[GRID_COLS * TILE_ROWS], float imgvf_inner[GRID_COLS * (TILE_ROWS + 2)], float I_inner[GRID_COLS * TILE_ROWS], int k)
{
#pragma HLS inline off
    if (flag) lc_mgvf(result_inner, imgvf_inner, I_inner, k);
    return;
}
}

extern "C"{
void buffer_store(int flag, int k, float *result_dest, float result_src[GRID_COLS * TILE_ROWS])
{
#pragma HLS inline off
    if (flag) memcpy(result_dest + k * TILE_ROWS * GRID_COLS, result_src, sizeof(float) * TILE_ROWS * GRID_COLS);
    return;
}
}

extern "C" {

__kernel void workload(float result[GRID_ROWS * GRID_COLS], float imgvf[GRID_ROWS * GRID_COLS], float I[GRID_ROWS * GRID_COLS])
{

    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=result
    #pragma HLS INTERFACE m_axi port=imgvf offset=slave bundle=imgvf
    #pragma HLS INTERFACE m_axi port=I offset=slave bundle=I
    
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=imgvf bundle=control
    #pragma HLS INTERFACE s_axilite port=I bundle=control
    
    #pragma HLS INTERFACE s_axilite port=return bundle=control
  
    float result_inner_0 [TILE_ROWS * GRID_COLS];
#pragma HLS array_partition variable=result_inner_0 cyclic factor=16
    float imgvf_inner_0   [(TILE_ROWS + 2) * GRID_COLS];
#pragma HLS array_partition variable=imgvf_inner_0   cyclic factor=16
    float I_inner_0  [TILE_ROWS * GRID_COLS];
#pragma HLS array_partition variable=I_inner_0  cyclic factor=16

    float result_inner_1 [TILE_ROWS * GRID_COLS];
#pragma HLS array_partition variable=result_inner_1 cyclic factor=16
    float imgvf_inner_1   [(TILE_ROWS + 2) * GRID_COLS];
#pragma HLS array_partition variable=imgvf_inner_1   cyclic factor=16
    float I_inner_1  [TILE_ROWS * GRID_COLS];
#pragma HLS array_partition variable=I_inner_1  cyclic factor=16

    float result_inner_2 [TILE_ROWS * GRID_COLS];
#pragma HLS array_partition variable=result_inner_2 cyclic factor=16
    float imgvf_inner_2   [(TILE_ROWS + 2) * GRID_COLS];
#pragma HLS array_partition variable=imgvf_inner_2   cyclic factor=16
    float I_inner_2  [TILE_ROWS * GRID_COLS];
#pragma HLS array_partition variable=I_inner_2  cyclic factor=16

    float result_inner_3 [TILE_ROWS * GRID_COLS];
#pragma HLS array_partition variable=result_inner_3 cyclic factor=16
    float imgvf_inner_3   [(TILE_ROWS + 2) * GRID_COLS];
#pragma HLS array_partition variable=imgvf_inner_3   cyclic factor=16
    float I_inner_3  [TILE_ROWS * GRID_COLS];
#pragma HLS array_partition variable=I_inner_3  cyclic factor=16

    int i , r , c;
    int k;

    for (i = 0; i < ITERATION/2; i++) {

        for (k = 0; k < GRID_ROWS / TILE_ROWS + 3; k++) {
            int load_imgvf_flag = k >= 0 && k < GRID_ROWS / TILE_ROWS;
            int load_I_flag = k >= 1 && k < GRID_ROWS / TILE_ROWS + 1;
            int compute_flag = k >= 2 && k < GRID_ROWS / TILE_ROWS + 2;
            int store_flag = k >= 3 && k < GRID_ROWS / TILE_ROWS + 3;
            
            if (k % 4 == 0) {
                buffer_load_imgvf(load_imgvf_flag, k, imgvf_inner_0, imgvf);
                buffer_load_I(load_I_flag, k - 1, I_inner_3, I);
                buffer_compute(compute_flag, result_inner_2, imgvf_inner_2, I_inner_2, k - 2);
                buffer_store(store_flag, k - 3, result, result_inner_1);
            }

            else if (k % 4 == 1) {
                buffer_load_imgvf(load_imgvf_flag, k, imgvf_inner_1, imgvf);
                buffer_load_I(load_I_flag, k - 1, I_inner_0, I);
                buffer_compute(compute_flag, result_inner_3, imgvf_inner_3, I_inner_3, k - 2);
                buffer_store(store_flag, k - 3, result, result_inner_2);
            }
            
            else if (k % 4 ==2){
                buffer_load_imgvf(load_imgvf_flag, k, imgvf_inner_2, imgvf);
                buffer_load_I(load_I_flag, k - 1, I_inner_1, I);
                buffer_compute(compute_flag, result_inner_0, imgvf_inner_0, I_inner_0, k - 2);
                buffer_store(store_flag, k - 3, result, result_inner_3);
            }
            else {
                buffer_load_imgvf(load_imgvf_flag, k, imgvf_inner_3, imgvf);
                buffer_load_I(load_I_flag, k - 1, I_inner_2, I);
                buffer_compute(compute_flag, result_inner_1, imgvf_inner_1, I_inner_1, k - 2);
                buffer_store(store_flag, k - 3, result, result_inner_0);
            }
        }

        for (k = 0; k < GRID_ROWS / TILE_ROWS + 3; k++) {
            int load_imgvf_flag = k >= 0 && k < GRID_ROWS / TILE_ROWS;
            int load_I_flag = k >= 1 && k < GRID_ROWS / TILE_ROWS + 1;
            int compute_flag = k >= 2 && k < GRID_ROWS / TILE_ROWS + 2;
            int store_flag = k >= 3 && k < GRID_ROWS / TILE_ROWS + 3;
            
            if (k % 4 == 0) {
                buffer_load_imgvf(load_imgvf_flag, k, imgvf_inner_0, result);
                buffer_load_I(load_I_flag, k - 1, I_inner_3, I);
                buffer_compute(compute_flag, result_inner_2, imgvf_inner_2, I_inner_2, k - 2);
                buffer_store(store_flag, k - 3, imgvf, result_inner_1);
            }

            else if (k % 4 == 1) {
                buffer_load_imgvf(load_imgvf_flag, k, imgvf_inner_1, result);
                buffer_load_I(load_I_flag, k - 1, I_inner_0, I);
                buffer_compute(compute_flag, result_inner_3, imgvf_inner_3, I_inner_3, k - 2);
                buffer_store(store_flag, k - 3, imgvf, result_inner_2);
            }
            
            else if (k % 4 ==2){
                buffer_load_imgvf(load_imgvf_flag, k, imgvf_inner_2, result);
                buffer_load_I(load_I_flag, k - 1, I_inner_1, I);
                buffer_compute(compute_flag, result_inner_0, imgvf_inner_0, I_inner_0, k - 2);
                buffer_store(store_flag, k - 3, imgvf, result_inner_3);
            }
            else {
                buffer_load_imgvf(load_imgvf_flag, k, imgvf_inner_3, result);
                buffer_load_I(load_I_flag, k - 1, I_inner_2, I);
                buffer_compute(compute_flag, result_inner_1, imgvf_inner_1, I_inner_1, k - 2);
                buffer_store(store_flag, k - 3, imgvf, result_inner_0);
            }

        }

    }

    return;
}
}