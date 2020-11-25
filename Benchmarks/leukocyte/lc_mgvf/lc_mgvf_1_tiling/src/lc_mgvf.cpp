#include"lc_mgvf.h"

extern "C" {

float heaviside(float x) {
    // A simpler, faster approximation of the Heaviside function
    float out = 0.0;
    if (x > -0.0001) out = 0.5;
    if (x >  0.0001) out = 1.0;
    return out; 
}

void lc_mgvf(float result[TILE_ROWS * GRID_COLS], float imgvf[(TILE_ROWS + 2) * GRID_COLS], float I[TILE_ROWS * GRID_COLS], int which_boundary)
{
    for (int i = 0; i < TILE_ROWS; i++) {
        for (int j = 0; j < GRID_COLS; j++) {
            float old_val = imgvf[i * GRID_COLS + j + GRID_COLS];

            float UL    = ((i == 0 && which_boundary == TOP              ||  j == 0              ) ? 0 : (imgvf[(i - 1   ) * GRID_COLS + (j - 1  ) + GRID_COLS] - old_val));
            float U     = ((i == 0 && which_boundary == TOP                                      ) ? 0 : (imgvf[(i - 1   ) * GRID_COLS + (j      ) + GRID_COLS] - old_val));
            float UR    = ((i == 0 && which_boundary == TOP              ||  j == GRID_COLS - 1  ) ? 0 : (imgvf[(i - 1   ) * GRID_COLS + (j + 1  ) + GRID_COLS] - old_val));

            float L     = ((                        j == 0              ) ? 0 : (imgvf[(i       ) * GRID_COLS + (j - 1  ) + GRID_COLS] - old_val));
            float R     = ((                        j == GRID_COLS - 1  ) ? 0 : (imgvf[(i       ) * GRID_COLS + (j + 1  ) + GRID_COLS] - old_val));

            float DL    = (((i == TILE_ROWS - 1)  && which_boundary == BOTTOM ||  j == 0              ) ? 0 : (imgvf[(i + 1   ) * GRID_COLS + (j - 1  ) + GRID_COLS] - old_val));
            float D     = (((i == TILE_ROWS - 1)  && which_boundary == BOTTOM                         ) ? 0 : (imgvf[(i + 1   ) * GRID_COLS + (j      ) + GRID_COLS] - old_val));
            float DR    = (((i == TILE_ROWS - 1)  && which_boundary == BOTTOM ||  j == GRID_COLS - 1  ) ? 0 : (imgvf[(i + 1   ) * GRID_COLS + (j + 1  ) + GRID_COLS] - old_val));

            float vHe = old_val + MU_O_LAMBDA * (heaviside(UL) * UL + heaviside(U) * U + heaviside(UR) * UR + heaviside(L) * L + heaviside(R) * R + heaviside(DL) * DL + heaviside(D) * D + heaviside(DR) * DR);

            float vI = I[i * GRID_COLS + j];
            float new_val = vHe - (ONE_O_LAMBDA * vI * (vHe - vI));
            result[i * GRID_COLS + j] = new_val;

        }
    }
}

void workload(float result[GRID_ROWS * GRID_COLS], float imgvf[GRID_ROWS * GRID_COLS], float I[GRID_ROWS * GRID_COLS])
{
    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=result1
    #pragma HLS INTERFACE m_axi port=imgvf offset=slave bundle=imgvf1
    #pragma HLS INTERFACE m_axi port=I offset=slave bundle=I1
    
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=imgvf bundle=control
    #pragma HLS INTERFACE s_axilite port=I bundle=control
    
    #pragma HLS INTERFACE s_axilite port=return bundle=control

	float result_inner [TILE_ROWS * GRID_COLS];
	float imgvf_inner [(TILE_ROWS + 2) * GRID_COLS];
	float I_inner [TILE_ROWS * GRID_COLS];

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
