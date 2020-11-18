#include"hotspot.h"

// #define PARA_FACTOR 1

extern "C" {

float hotspot_stencil_core(float temp_top, float temp_left, float temp_right, float temp_bottom, float temp_center, float power_center, float Cap_1, float Rx_1, float Ry_1, float Rz_1) {
    #pragma HLS inline off
    float tmp = (float)temp_center + (float)temp_center;
	float tmp0 = (float)temp_top + (float)temp_bottom - (float)tmp;
	float tmp1 = (float)temp_left + (float)temp_right - (float)tmp;
	float tmp2 = (float)AMB_TEMP - (float)temp_center;
	power_center += (float)(tmp0 * (float)Ry_1);
	power_center += (float)(tmp1 * (float)Rx_1);
	power_center += (float)(tmp2 * (float)Rz_1);
	temp_center += (float)Cap_1 * power_center;
	return temp_center;
}

void hotspot(float result[TILE_ROWS * GRID_COLS], float temp[(TILE_ROWS + 2) * GRID_COLS], float power[TILE_ROWS * GRID_COLS], float Cap_1, float Rx_1, float Ry_1, float Rz_1, int which_boundary) {
    float delta;
    int c, r, i, j, k, ii;
    float temp_top[PARA_FACTOR], temp_left[PARA_FACTOR], temp_right[PARA_FACTOR], temp_bottom[PARA_FACTOR], temp_center[PARA_FACTOR], power_center[PARA_FACTOR];

    float temp_rf [PARA_FACTOR][GRID_COLS * 2 / PARA_FACTOR + 1];
    #pragma HLS array_partition variable=temp_rf complete dim=0

    for (i = 0 ; i < GRID_COLS * 2 / PARA_FACTOR + 1; i++) {
        #pragma HLS pipeline II=1
        for (ii = 0; ii < PARA_FACTOR; ii++) {
            #pragma HLS unroll
            temp_rf[ii][i] = temp[i*PARA_FACTOR + ii];
        }
    }

    for (i = 0; i < GRID_COLS / PARA_FACTOR * TILE_ROWS  ; i++) {
        #pragma HLS pipeline II=1
        for (k = 0; k < PARA_FACTOR; k++) {
            #pragma HLS unroll
            temp_center[k]  = temp_rf[k][GRID_COLS / PARA_FACTOR];
      
            temp_top[k]     = (i < GRID_COLS / PARA_FACTOR && which_boundary == TOP) ? temp_center[k] : temp_rf[k][0];

            temp_left[k]    = ((i % (GRID_COLS / PARA_FACTOR)) == 0 && k == 0) ? temp_center[k] : temp_rf[(k - 1 + PARA_FACTOR) % PARA_FACTOR][GRID_COLS / PARA_FACTOR - (k == 0) ];

            temp_right[k]   = ((i % (GRID_COLS / PARA_FACTOR)) == (GRID_COLS / PARA_FACTOR - 1) && k == PARA_FACTOR - 1) ? temp_center[k] : temp_rf[(k + 1 + PARA_FACTOR) % PARA_FACTOR][GRID_COLS / PARA_FACTOR + (k == (PARA_FACTOR - 1)) ];

            temp_bottom[k]  = (i >= GRID_COLS / PARA_FACTOR * (TILE_ROWS - 1) && which_boundary == BOTTOM) ? temp_center[k] : temp_rf[k][GRID_COLS / PARA_FACTOR * 2];

            power_center[k] = power[i * PARA_FACTOR + k];

            result[i * PARA_FACTOR + k] = hotspot_stencil_core(temp_top[k], temp_left[k], temp_right[k], temp_bottom[k], temp_center[k], power_center[k], Cap_1, Rx_1, Ry_1, Rz_1);
        }

        for (k = 0; k < PARA_FACTOR; k++) {
            #pragma hls unroll
            for (j = 0; j < GRID_COLS * 2 / PARA_FACTOR; j++) {
                #pragma hls unroll
                temp_rf[k][j] = temp_rf[k][j + 1];
            }

            temp_rf[k][GRID_COLS * 2 / PARA_FACTOR] = temp[GRID_COLS * 2 + (i + 1) * PARA_FACTOR + k];
        }
    }

    return;
}

void workload(float result[GRID_ROWS * GRID_COLS], float temp[GRID_ROWS * GRID_COLS], float power[GRID_ROWS * GRID_COLS]){

    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=temp offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=power offset=slave bundle=gmem2
    
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=temp bundle=control
    #pragma HLS INTERFACE s_axilite port=power bundle=control
    
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    float result_inner [TILE_ROWS * GRID_COLS];
    float temp_inner   [(TILE_ROWS + 2) * GRID_COLS];
    float power_inner  [TILE_ROWS * GRID_COLS];

    float grid_height = CHIP_HEIGHT / GRID_ROWS;
    float grid_width = CHIP_WIDTH / GRID_COLS;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * T_CHIP * grid_width * grid_height;
    float Rx = grid_width / (2.0 * K_SI * T_CHIP * grid_height);
    float Ry = grid_height / (2.0 * K_SI * T_CHIP * grid_width);
    float Rz = T_CHIP / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * T_CHIP * SPEC_HEAT_SI);
    float step = PRECISION / max_slope / 1000.0;

    float Rx_1=1.f / Rx;
    float Ry_1=1.f / Ry;
    float Rz_1=1.f / Rz;
    float Cap_1 = step / Cap;

    int i , r , c;
    int k;

    for (i = 0; i < SIM_TIME/2; i++) {
        for (k = 0; k < GRID_ROWS / TILE_ROWS; k++) {
            memcpy(temp_inner, temp + k * TILE_ROWS * GRID_COLS - GRID_COLS, sizeof(float) * (TILE_ROWS + 2) * GRID_COLS);
            memcpy(power_inner, power + k * TILE_ROWS * GRID_COLS, sizeof(float) * TILE_ROWS * GRID_COLS); 

            hotspot(result_inner, temp_inner, power_inner, Cap_1, Rx_1, Ry_1, Rz_1, k);

            memcpy(result + k * TILE_ROWS * GRID_COLS, result_inner, sizeof(float) * TILE_ROWS * GRID_COLS);
        }

        for (k = 0; k < GRID_ROWS / TILE_ROWS; k++) {
            memcpy(temp_inner, result + k * TILE_ROWS * GRID_COLS - GRID_COLS, sizeof(float) * (TILE_ROWS + 2) * GRID_COLS);
            memcpy(power_inner, power + k * TILE_ROWS * GRID_COLS, sizeof(float) * TILE_ROWS * GRID_COLS); 

            hotspot(result_inner, temp_inner, power_inner, Cap_1, Rx_1, Ry_1, Rz_1, k);

            memcpy(temp + k * TILE_ROWS * GRID_COLS, result_inner,  sizeof(float) * TILE_ROWS * GRID_COLS);
        }

    }

    return;
}
}
