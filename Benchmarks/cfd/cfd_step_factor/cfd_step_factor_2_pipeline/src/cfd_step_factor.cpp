#include"cfd_step_factor.h"

#define PARA_FACTOR 1

extern "C" {
void compute_velocity(float& density, float3& momentum, float3& velocity)
{
#pragma HLS inline off
    velocity.x = (float)momentum.x / (float)density;
    velocity.y = (float)momentum.y / (float)density;
    velocity.z = (float)momentum.z / (float)density;
}

float compute_speed_sqd(float3& velocity)
{
#pragma HLS inline off
    return (float)velocity.x*(float)velocity.x + (float)velocity.y*(float)velocity.y + (float)velocity.z*(float)velocity.z;
}

float compute_pressure(float& density, float& density_energy, float& speed_sqd)
{
#pragma HLS inline off
    return (float(GAMMA) - float(1.0f))*(float)((float)density_energy - float(0.5f)*(float)((float)density*(float)speed_sqd));
}

float compute_speed_of_sound(float& density, float& pressure)
{
#pragma HLS inline off
    return (float)sqrt((float)(float(GAMMA)*(float)pressure / (float)density));
}

void cfd_step_factor(float result[TILE_ROWS], float variables[TILE_ROWS * NVAR], float areas[TILE_ROWS])
{
#pragma HLS inline off
    for (int i = 0; i < TILE_ROWS / PARA_FACTOR; i++) {
#pragma HLS pipeline II=1

        float density[PARA_FACTOR];
        float3 momentum[PARA_FACTOR];
        float density_energy[PARA_FACTOR];

        int iii = i * PARA_FACTOR;

        density[0] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 0 ]; 
        momentum[0].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 0)];
        momentum[0].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 0)];
        momentum[0].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 0)];
        density_energy[0] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 0 ];

        for (int ii = 0; ii < PARA_FACTOR; ii++) {
#pragma HLS unroll
            int iii = i * PARA_FACTOR + ii;
            float3 velocity;       compute_velocity(density[ii], momentum[ii], velocity);
            float speed_sqd = compute_speed_sqd(velocity);
            float pressure = compute_pressure(density[ii], density_energy[ii], speed_sqd);
            float speed_of_sound = compute_speed_of_sound(density[ii], pressure);
        
                result[iii] = float(0.5f) / (float)((float)sqrt((float)areas[iii]) * (float)((float)sqrt((float)speed_sqd) + (float)speed_of_sound));
        }
    }
}

void workload(float result[SIZE], float variables[SIZE * NVAR], float areas[SIZE])
{
    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=result
    #pragma HLS INTERFACE m_axi port=variables offset=slave bundle=variables
    #pragma HLS INTERFACE m_axi port=areas offset=slave bundle=areas
    
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=variables bundle=control
    #pragma HLS INTERFACE s_axilite port=areas bundle=control
    
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    float result_inner      [TILE_ROWS];
    float variables_inner   [TILE_ROWS * NVAR];
    float areas_inner       [TILE_ROWS];

    for (int k = 0; k < SIZE / TILE_ROWS; k++){
        
        memcpy(variables_inner, variables + k * TILE_ROWS * NVAR, sizeof(float) * TILE_ROWS * NVAR);
        memcpy(areas_inner, areas + k * TILE_ROWS, sizeof(float) * TILE_ROWS);

        cfd_step_factor(result_inner, variables_inner, areas_inner);

        memcpy(result + k * TILE_ROWS, result_inner, sizeof(float) * TILE_ROWS);
    }
    
    return;
}

}

