#include"cfd_step_factor.h"

extern "C" {

inline void compute_velocity(float& density, float3& momentum, float3& velocity)
{
    velocity.x = momentum.x / density;
    velocity.y = momentum.y / density;
    velocity.z = momentum.z / density;
}

inline float compute_speed_sqd(float3& velocity)
{
    return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}

inline float compute_pressure(float& density, float& density_energy, float& speed_sqd)
{
    return (float(GAMMA) - float(1.0f))*(density_energy - float(0.5f)*density*speed_sqd);
}

inline float compute_speed_of_sound(float& density, float& pressure)
{
    return sqrt(float(GAMMA)*pressure / density);
}

void cfd_step_factor(float result[TILE_ROWS], float variables[TILE_ROWS * NVAR], float areas[TILE_ROWS])
{
    for (int i = 0; i < TILE_ROWS; i++)
    {
     float density = variables[NVAR*i + VAR_DENSITY];

     float3 momentum;
     momentum.x = variables[NVAR*i + (VAR_MOMENTUM + 0)];
     momentum.y = variables[NVAR*i + (VAR_MOMENTUM + 1)];
     momentum.z = variables[NVAR*i + (VAR_MOMENTUM + 2)];

     float density_energy = variables[NVAR*i + VAR_DENSITY_ENERGY];
     float3 velocity;       compute_velocity(density, momentum, velocity);
     float speed_sqd = compute_speed_sqd(velocity);
     float pressure = compute_pressure(density, density_energy, speed_sqd);
     float speed_of_sound = compute_speed_of_sound(density, pressure);

     result[i] = float(0.5f) / (sqrt(areas[i]) * (sqrt(speed_sqd) + speed_of_sound));
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

