#include "hotspot.h"
#include "../../../common/mc.h"

extern "C"{
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
    
    for (i = 0; i < GRID_COLS / PARA_FACTOR * TILE_ROWS; i++) {
        #pragma HLS pipeline II=1
        for (k = 0; k < PARA_FACTOR; k++) {
            #pragma HLS unroll
            temp_center[k]  = temp_rf[k][GRID_COLS / PARA_FACTOR];
            temp_top[k]     = (i < GRID_COLS / PARA_FACTOR && which_boundary == TOP) ? temp_center[k] : temp_rf[k][0];
            temp_left[k]    = ((i % (GRID_COLS / PARA_FACTOR)) == 0 && k == 0) ? temp_center[k] : temp_rf[(k - 1 + PARA_FACTOR) % PARA_FACTOR][GRID_COLS / PARA_FACTOR - (k == 0) ];
            temp_right[k]   = ((i % (GRID_COLS / PARA_FACTOR)) == (GRID_COLS / PARA_FACTOR - 1) && k == PARA_FACTOR - 1) ? 
                                temp_center[k] : temp_rf[(k + 1 + PARA_FACTOR) % PARA_FACTOR][GRID_COLS / PARA_FACTOR + (k == (PARA_FACTOR - 1)) ];
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

void buffer_load(int flag, int k, float temp_dest[GRID_COLS * (TILE_ROWS + 2)], class ap_uint<LARGE_BUS> *temp_src, float power_dest[GRID_COLS * TILE_ROWS], class ap_uint<LARGE_BUS> *power_src)
{
    if (flag) {
		memcpy_wide_bus_read_float(temp_dest, (class ap_uint<LARGE_BUS> *)(temp_src + (k * TILE_ROWS * GRID_COLS - GRID_COLS) / (LARGE_BUS / 32)),0 * sizeof(float) , sizeof(float) *((unsigned long) ((TILE_ROWS + 2) * GRID_COLS)) );
		memcpy_wide_bus_read_float(power_dest, (class ap_uint<LARGE_BUS> *)(power_src + (k * TILE_ROWS * GRID_COLS) / (LARGE_BUS / 32)),0 * sizeof(float) , sizeof(float) *((unsigned long) (TILE_ROWS * GRID_COLS)) );
	}
    return;
}

void buffer_compute(int flag, float result_inner[GRID_COLS * TILE_ROWS], float temp_inner[GRID_COLS * (TILE_ROWS + 2)], float power_inner[GRID_COLS * TILE_ROWS], float Cap_1, float Rx_1, float Ry_1, float Rz_1, int k){
    if (flag) hotspot(result_inner, temp_inner, power_inner, Cap_1, Rx_1, Ry_1, Rz_1, k);
    return;
}

void buffer_store(int flag, int k, class ap_uint<LARGE_BUS> *result_dest, float result_src[GRID_COLS * TILE_ROWS])
{
    if (flag) memcpy_wide_bus_write_float((class ap_uint<LARGE_BUS> *)(result_dest + (k * TILE_ROWS * GRID_COLS) / (LARGE_BUS / 32)), result_src,0 * sizeof(float) , sizeof(float) *((unsigned long) (TILE_ROWS * GRID_COLS)) );
    return;
}

void workload(class ap_uint<LARGE_BUS> *result, class ap_uint<LARGE_BUS> *temp, class ap_uint<LARGE_BUS> *power)
{
    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=result
    #pragma HLS INTERFACE m_axi port=temp offset=slave bundle=temp
    #pragma HLS INTERFACE m_axi port=power offset=slave bundle=power
    
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=temp bundle=control
    #pragma HLS INTERFACE s_axilite port=power bundle=control
    
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    float result_inner_0 [TILE_ROWS * GRID_COLS];
    #pragma HLS array_partition variable=result_inner_0 cyclic factor=16
    float temp_inner_0   [(TILE_ROWS + 2) * GRID_COLS];
    #pragma HLS array_partition variable=temp_inner_0   cyclic factor=16
    float power_inner_0  [TILE_ROWS * GRID_COLS];
    #pragma HLS array_partition variable=power_inner_0  cyclic factor=16

    float result_inner_1 [TILE_ROWS * GRID_COLS];
    #pragma HLS array_partition variable=result_inner_1 cyclic factor=16
    float temp_inner_1   [(TILE_ROWS + 2) * GRID_COLS];
    #pragma HLS array_partition variable=temp_inner_1   cyclic factor=16
    float power_inner_1  [TILE_ROWS * GRID_COLS];
    #pragma HLS array_partition variable=power_inner_1  cyclic factor=16

    float result_inner_2 [TILE_ROWS * GRID_COLS];
    #pragma HLS array_partition variable=result_inner_2 cyclic factor=16
    float temp_inner_2   [(TILE_ROWS + 2) * GRID_COLS];
    #pragma HLS array_partition variable=temp_inner_2   cyclic factor=16
    float power_inner_2  [TILE_ROWS * GRID_COLS];
    #pragma HLS array_partition variable=power_inner_2  cyclic factor=16

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
        for (k = 0; k < GRID_ROWS / TILE_ROWS + 2; k++) {
            int load_flag = k >= 0 && k < GRID_ROWS / TILE_ROWS;
            int compute_flag = k >= 1 && k < GRID_ROWS / TILE_ROWS + 1;
            int store_flag = k >= 2 && k < GRID_ROWS / TILE_ROWS + 2;
            
            if (k % 3 == 0) {
                buffer_load(load_flag, k, temp_inner_0, temp, power_inner_0, power);
                buffer_compute(compute_flag, result_inner_2, temp_inner_2, power_inner_2, Cap_1, Rx_1, Ry_1, Rz_1, k - 1);
                buffer_store(store_flag, k - 2, result, result_inner_1);
            } else if (k % 3 == 1) {
                buffer_load(load_flag, k, temp_inner_1, temp, power_inner_1, power);
                buffer_compute(compute_flag, result_inner_0, temp_inner_0, power_inner_0, Cap_1, Rx_1, Ry_1, Rz_1, k - 1);
                buffer_store(store_flag, k - 2, result, result_inner_2);
            } else {
                buffer_load(load_flag, k, temp_inner_2, temp, power_inner_2, power);
                buffer_compute(compute_flag, result_inner_1, temp_inner_1, power_inner_1, Cap_1, Rx_1, Ry_1, Rz_1, k - 1);
                buffer_store(store_flag, k - 2, result, result_inner_0);
            }
        }

        for (k = 0; k < GRID_ROWS / TILE_ROWS + 2; k++) {
            int load_flag = k >= 0 && k < GRID_ROWS / TILE_ROWS;
            int compute_flag = k >= 1 && k < GRID_ROWS / TILE_ROWS + 1;
            int store_flag = k >= 2 && k < GRID_ROWS / TILE_ROWS + 2;
            
            if (k % 3 == 0) {
                buffer_load(load_flag, k, temp_inner_0, result, power_inner_0, power);
                buffer_compute(compute_flag, result_inner_2, temp_inner_2, power_inner_2, Cap_1, Rx_1, Ry_1, Rz_1, k - 1);
                buffer_store(store_flag, k - 2, temp, result_inner_1);
            } else if (k % 3 == 1) {
                buffer_load(load_flag, k, temp_inner_1, result, power_inner_1, power);
                buffer_compute(compute_flag, result_inner_0, temp_inner_0, power_inner_0, Cap_1, Rx_1, Ry_1, Rz_1, k - 1);
                buffer_store(store_flag, k - 2, temp, result_inner_2);
            } else {
                buffer_load(load_flag, k, temp_inner_2, result, power_inner_2, power);
                buffer_compute(compute_flag, result_inner_1, temp_inner_1, power_inner_1, Cap_1, Rx_1, Ry_1, Rz_1, k - 1);
                buffer_store(store_flag, k - 2, temp, result_inner_0);
            }
        }
    }

    return;
}
}
