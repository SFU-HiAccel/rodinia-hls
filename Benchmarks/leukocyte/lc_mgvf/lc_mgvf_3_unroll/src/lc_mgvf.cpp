#include "lc_mgvf.h"

extern "C" {

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





void workload(float result[GRID_ROWS * GRID_COLS], float imgvf[GRID_ROWS * GRID_COLS], float I[GRID_ROWS * GRID_COLS])
{

    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=result
    #pragma HLS INTERFACE m_axi port=imgvf offset=slave bundle=imgvf
    #pragma HLS INTERFACE m_axi port=I offset=slave bundle=I
    
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=imgvf bundle=control
    #pragma HLS INTERFACE s_axilite port=I bundle=control
    
    #pragma HLS INTERFACE s_axilite port=return bundle=control

	float result_inner [TILE_ROWS * GRID_COLS];
#pragma HLS array_partition variable=result_inner cyclic factor=16
	float imgvf_inner [(TILE_ROWS + 2) * GRID_COLS];
#pragma HLS array_partition variable=imgvf_inner cyclic factor=16
	float I_inner [TILE_ROWS * GRID_COLS];
#pragma HLS array_partition variable=I_inner  cyclic factor=16



    
    	int i;
    	float diff = 1.0;
    	for (i = 0; i < ITERATION / 2; i++) {
		int k;
		for(k = 0; k < GRID_ROWS / TILE_ROWS; k++) {

			memcpy(imgvf_inner, imgvf + k * TILE_ROWS * GRID_COLS - GRID_COLS, sizeof(float) * (TILE_ROWS + 2) * GRID_COLS);
			memcpy(I_inner, I + k * TILE_ROWS * GRID_COLS, sizeof(float) * TILE_ROWS * GRID_COLS);

        		lc_mgvf(result_inner, imgvf_inner, I_inner, k);

			memcpy(result + k * TILE_ROWS * GRID_COLS, result_inner, sizeof(float) * TILE_ROWS * GRID_COLS);

		}

		for(k = 0; k < GRID_ROWS / TILE_ROWS; k++) {

			memcpy(imgvf_inner, result + k * TILE_ROWS * GRID_COLS - GRID_COLS, sizeof(float) * (TILE_ROWS + 2) * GRID_COLS);
			memcpy(I_inner, I + k * TILE_ROWS * GRID_COLS, sizeof(float) * TILE_ROWS * GRID_COLS);

        		lc_mgvf(result_inner, imgvf_inner, I_inner, k);

			memcpy(imgvf + k * TILE_ROWS * GRID_COLS, result_inner, sizeof(float) * TILE_ROWS * GRID_COLS);

		}

    	}

    return;

}








}
