#include "lc_mgvf.h"
#include "../../../../common/mc.h"

#define __kernel
#define __global

float heaviside(float x) {
    // A simpler, faster approximation of the Heaviside function
    float out = 0.0;
    if ((float)x > (float)-0.0001) out = (float)0.5;
    if ((float)x >  (float)0.0001) out = (float)1.0;
    return out; 
}

float lc_mgvf_stencil_core(float c, float ul, float u, float ur, float l, float r, float dl, float d, float dr, float vI)
{
    float UL = (float)ul - (float)c;
    float U  = (float)u  - (float)c;
    float UR = (float)ur - (float)c;
    float L  = (float)l  - (float)c;
    float R  = (float)r  - (float)c;
    float DL = (float)dl - (float)c;
    float D  = (float)d  - (float)c;
    float DR = (float)dr - (float)c;

    float vHe = (float)c + (float)MU_O_LAMBDA * (float)(
                                   (float)heaviside(UL) * (float)UL + 
                                   (float)heaviside(U)  * (float)U  + 
                                   (float)heaviside(UR) * (float)UR + 
                                   (float)heaviside(L)  * (float)L  + 
                                   (float)heaviside(R)  * (float)R  + 
                                   (float)heaviside(DL) * (float)DL + 
                                   (float)heaviside(D)  * (float)D  + 
                                   (float)heaviside(DR) * (float)DR 
                                   );

    float new_val = (float)vHe - ((float)ONE_O_LAMBDA * (float)vI * (float)((float)vHe - (float)vI));

    return new_val;
}

void lc_mgvf(float result[TILE_ROWS * GRID_COLS], float imgvf[(TILE_ROWS + 2) * GRID_COLS], float I[TILE_ROWS * GRID_COLS], int which_boundary)
{
	int cols = GRID_COLS;
	int rows = GRID_ROWS;
	float imgvf_rf[GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS * 2 + PARA_FACTOR];
#pragma HLS array_partition variable=imgvf_rf cyclic dim=0 factor=64

	int i;

    for (i = 0; i < GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR; i++) {
#pragma HLS pipeline II=1
#pragma HLS unroll
        imgvf_rf[i + MAX_RADIUS] = imgvf[i];
    }


	//for (i = -(GRID_COLS * (2 * MAX_RADIUS) + MAX_RADIUS + PARA_FACTOR) / PARA_FACTOR; i < GRID_COLS / PARA_FACTOR * TILE_ROWS; i++) {
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
void buffer_load(int flag, int k, float imgvf_dest[GRID_COLS * (TILE_ROWS + 2)], class ap_uint<LARGE_BUS> *imgvf_src, float I_dest[GRID_COLS * TILE_ROWS], class ap_uint<LARGE_BUS> *I_src)
{
#pragma HLS inline off
    if (flag) {
		memcpy_wide_bus_read_float(imgvf_dest, (class ap_uint<LARGE_BUS> *)(imgvf_src + (k * TILE_ROWS * GRID_COLS - GRID_COLS) / (LARGE_BUS / 32)),0 * sizeof(float) , sizeof(float) *((unsigned long) ((TILE_ROWS + 2) * GRID_COLS)) );
		memcpy_wide_bus_read_float(I_dest, (class ap_uint<LARGE_BUS> *)(I_src + (k * TILE_ROWS * GRID_COLS) / (LARGE_BUS / 32)),0 * sizeof(float) , sizeof(float) *((unsigned long) (TILE_ROWS * GRID_COLS)) );
		}
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
void buffer_store(int flag, int k, class ap_uint<LARGE_BUS> *result_dest, float result_src[GRID_COLS * TILE_ROWS])
{
#pragma HLS inline off
    if (flag) memcpy_wide_bus_write_float((class ap_uint<LARGE_BUS> *)(result_dest + (k * TILE_ROWS * GRID_COLS) / (LARGE_BUS / 32)), result_src,0 * sizeof(float) , sizeof(float) *((unsigned long) (TILE_ROWS * GRID_COLS)) );
    return;
}
}

extern "C" {
__kernel void workload(class ap_uint<LARGE_BUS> *result, class ap_uint<LARGE_BUS> *imgvf, class ap_uint<LARGE_BUS> *I)
{
    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=result1
    #pragma HLS INTERFACE m_axi port=imgvf offset=slave bundle=imgvf1
    #pragma HLS INTERFACE m_axi port=I offset=slave bundle=I1
    
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

    int i , r , c;
    int k;

    for (i = 0; i < ITERATION/2; i++) {

        for (k = 0; k < GRID_ROWS / TILE_ROWS + 2; k++) {
            int load_flag = k >= 0 && k < GRID_ROWS / TILE_ROWS;
            int compute_flag = k >= 1 && k < GRID_ROWS / TILE_ROWS + 1;
            int store_flag = k >= 2 && k < GRID_ROWS / TILE_ROWS + 2;
            
            if (k % 3 == 0) {
                buffer_load(load_flag, k, imgvf_inner_0, imgvf, I_inner_0, I);
                buffer_compute(compute_flag, result_inner_2, imgvf_inner_2, I_inner_2, k - 1);
                buffer_store(store_flag, k - 2, result, result_inner_1);
            }

            else if (k % 3 == 1) {
                buffer_load(load_flag, k, imgvf_inner_1, imgvf, I_inner_1, I);
                buffer_compute(compute_flag, result_inner_0, imgvf_inner_0, I_inner_0, k - 1);
                buffer_store(store_flag, k - 2, result, result_inner_2);
            }
            
            else{
                buffer_load(load_flag, k, imgvf_inner_2, imgvf, I_inner_2, I);
                buffer_compute(compute_flag, result_inner_1, imgvf_inner_1, I_inner_1, k - 1);
                buffer_store(store_flag, k - 2, result, result_inner_0);
            }
        }

        for (k = 0; k < GRID_ROWS / TILE_ROWS + 2; k++) {
            int load_flag = k >= 0 && k < GRID_ROWS / TILE_ROWS;
            int compute_flag = k >= 1 && k < GRID_ROWS / TILE_ROWS + 1;
            int store_flag = k >= 2 && k < GRID_ROWS / TILE_ROWS + 2;
            
            if (k % 3 == 0) {
                buffer_load(load_flag, k, imgvf_inner_0, result, I_inner_0, I);
                buffer_compute(compute_flag, result_inner_2, imgvf_inner_2, I_inner_2, k - 1);
                buffer_store(store_flag, k - 2, imgvf, result_inner_1);
            }

            else if (k % 3 == 1) {
                buffer_load(load_flag, k, imgvf_inner_1, result, I_inner_1, I);
                buffer_compute(compute_flag, result_inner_0, imgvf_inner_0, I_inner_0, k - 1);
                buffer_store(store_flag, k - 2, imgvf, result_inner_2);
            }
            
            else{
                buffer_load(load_flag, k, imgvf_inner_2, result, I_inner_2, I);
                buffer_compute(compute_flag, result_inner_1, imgvf_inner_1, I_inner_1, k - 1);
                buffer_store(store_flag, k - 2, imgvf, result_inner_0);
            }
        }
    }

    return;
}
}
