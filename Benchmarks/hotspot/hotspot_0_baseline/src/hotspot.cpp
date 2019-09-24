#include"hotspot.h"

extern"C" {

void hotspot(float result[GRID_ROWS * GRID_COLS], float temp[GRID_ROWS * GRID_COLS], float power[GRID_ROWS * GRID_COLS], float Cap_1, float Rx_1, float Ry_1, float Rz_1) {
    float amb_temp = 80.0;
    float delta;

    for (int r = 0; r < GRID_ROWS; r++)
        for (int c = 0; c < GRID_COLS; c++) {
            if (r == 0 || c == 0 || r == GRID_ROWS - 1 || c == GRID_COLS - 1) {

                /* Corner 1 */
                if ((r == 0) && (c == 0)) {
                    delta = (Cap_1) * (power[0] +
                        (temp[1] - temp[0]) * Rx_1 +
                        (temp[GRID_COLS] - temp[0]) * Ry_1 +
                        (amb_temp - temp[0]) * Rz_1);
                }   
    
                /* Corner 2 */
                else if ((r == 0) && (c == GRID_COLS - 1)) {
                    delta = (Cap_1) * (power[c] +
                        (temp[c - 1] - temp[c]) * Rx_1 +
                        (temp[c + GRID_COLS] - temp[c]) * Ry_1 +
                        (amb_temp - temp[c]) * Rz_1);
                }   
    
                /* Corner 3 */
                else if ((r == GRID_ROWS - 1) && (c == GRID_COLS - 1)) {
                    delta = (Cap_1) * (power[r*GRID_COLS + c] +
                        (temp[r*GRID_COLS + c - 1] - temp[r*GRID_COLS + c]) * Rx_1 +
                        (temp[(r - 1)*GRID_COLS + c] - temp[r*GRID_COLS + c]) * Ry_1 +
                        (amb_temp - temp[r*GRID_COLS + c]) * Rz_1);
                }   
    
                /* Corner 4 */
                else if ((r == GRID_ROWS - 1) && (c == 0)) {
                    delta = (Cap_1) * (power[r*GRID_COLS] +
                        (temp[r*GRID_COLS + 1] - temp[r*GRID_COLS]) * Rx_1 +
                        (temp[(r - 1)*GRID_COLS] - temp[r*GRID_COLS]) * Ry_1 +
                        (amb_temp - temp[r*GRID_COLS]) * Rz_1);
                }   
    
                /* Edge 1 */
                else if (r == 0) {
                    delta = (Cap_1) * (power[c] +
                        (temp[c + 1] + temp[c - 1] - 2.0*temp[c]) * Rx_1 +
                        (temp[GRID_COLS + c] - temp[c]) * Ry_1 +
                        (amb_temp - temp[c]) * Rz_1);
                }   
    
                /* Edge 2 */
                else if (c == GRID_COLS - 1) {
                    delta = (Cap_1) * (power[r*GRID_COLS + c] +
                        (temp[(r + 1)*GRID_COLS + c] + temp[(r - 1)*GRID_COLS + c] - 2.0*temp[r*GRID_COLS + c]) * Ry_1 +
                        (temp[r*GRID_COLS + c - 1] - temp[r*GRID_COLS + c]) * Rx_1 +
                        (amb_temp - temp[r*GRID_COLS + c]) * Rz_1);
                }   
    
                /* Edge 3 */
                else if (r == GRID_ROWS - 1) {
                    delta = (Cap_1) * (power[r*GRID_COLS + c] +
                        (temp[r*GRID_COLS + c + 1] + temp[r*GRID_COLS + c - 1] - 2.0*temp[r*GRID_COLS + c]) * Rx_1 +
                        (temp[(r - 1)*GRID_COLS + c] - temp[r*GRID_COLS + c]) * Ry_1 +
                        (amb_temp - temp[r*GRID_COLS + c]) * Rz_1);
                }   
    
                /* Edge 4 */
                else if (c == 0) {
                    delta = (Cap_1) * (power[r*GRID_COLS] +
                        (temp[(r + 1)*GRID_COLS] + temp[(r - 1)*GRID_COLS] - 2.0*temp[r*GRID_COLS]) * Ry_1 +
                        (temp[r*GRID_COLS + 1] - temp[r*GRID_COLS]) * Rx_1 +
                        (amb_temp - temp[r*GRID_COLS]) * Rz_1);
                }

            }

            else {
                    delta = (Cap_1 * (power[r*GRID_COLS + c] +
                        (temp[(r + 1)*GRID_COLS + c] + temp[(r - 1)*GRID_COLS + c] - 2.f*temp[r*GRID_COLS + c]) * Ry_1 +
                        (temp[r*GRID_COLS + c + 1] + temp[r*GRID_COLS + c - 1] - 2.f*temp[r*GRID_COLS + c]) * Rx_1 +
                        (amb_temp - temp[r*GRID_COLS + c]) * Rz_1));
            }

            result[r*GRID_COLS + c] = temp[r*GRID_COLS + c] + delta;

        }


    return;
}




void workload(float result[GRID_ROWS * GRID_COLS], float temp[GRID_ROWS * GRID_COLS], float power[GRID_ROWS * GRID_COLS])
{

    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=temp offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=power offset=slave bundle=gmem
    
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=temp bundle=control
    #pragma HLS INTERFACE s_axilite port=power bundle=control
    
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
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

    int i;
    for (i = 0; i < SIM_TIME/2; i++) {
     
   hotspot(result, temp, power, Cap_1, Rx_1, Ry_1, Rz_1);
   
   hotspot(temp, result, power, Cap_1, Rx_1, Ry_1, Rz_1);

    }

return;


}

}
