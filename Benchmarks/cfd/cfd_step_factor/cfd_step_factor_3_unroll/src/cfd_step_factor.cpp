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

        density[1] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 5 ]; 
        momentum[1].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 5)];
        momentum[1].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 5)];
        momentum[1].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 5)];
        density_energy[1] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 5 ];


        density[2] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 10 ]; 
        momentum[2].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 10)];
        momentum[2].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 10)];
        momentum[2].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 10)];
        density_energy[2] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 10 ];


        density[3] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 15 ]; 
        momentum[3].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 15)];
        momentum[3].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 15)];
        momentum[3].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 15)];
        density_energy[3] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 15 ];


        density[4] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 20 ]; 
        momentum[4].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 20)];
        momentum[4].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 20)];
        momentum[4].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 20)];
        density_energy[4] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 20 ];


        density[5] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 25 ]; 
        momentum[5].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 25)];
        momentum[5].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 25)];
        momentum[5].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 25)];
        density_energy[5] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 25 ];


        density[6] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 30 ]; 
        momentum[6].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 30)];
        momentum[6].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 30)];
        momentum[6].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 30)];
        density_energy[6] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 30 ];


        density[7] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 35 ]; 
        momentum[7].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 35)];
        momentum[7].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 35)];
        momentum[7].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 35)];
        density_energy[7] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 35 ];



        density[8] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 40 ]; 
        momentum[8].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 40)];
        momentum[8].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 40)];
        momentum[8].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 40)];
        density_energy[8] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 40 ];

        density [9] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 45 ]; 
        momentum[9].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 45)];
        momentum[9].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 45)];
        momentum[9].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 45)];
        density_energy[9] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 45 ];


        density[10] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 50 ]; 
        momentum[10].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 50)];
        momentum[10].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 50)];
        momentum[10].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 50)];
        density_energy[10] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 50 ];


        density[11] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 55 ]; 
        momentum[11].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 55)];
        momentum[11].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 55)];
        momentum[11].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 55)];
        density_energy[11] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 55 ];


        density[12] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 60 ]; 
        momentum[12].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 60)];
        momentum[12].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 60)];
        momentum[12].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 60)];
        density_energy[12] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 60 ];


        density[13] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 65 ]; 
        momentum[13].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 65)];
        momentum[13].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 65)];
        momentum[13].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 65)];
        density_energy[13] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 65 ];


        density[14] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 70 ]; 
        momentum[14].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 70)];
        momentum[14].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 70)];
        momentum[14].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 70)];
        density_energy[14] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 70 ];


        density[15] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 75 ]; 
        momentum[15].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 75)];
        momentum[15].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 75)];
        momentum[15].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 75)];
        density_energy[15] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 75 ];

        for (int ii = 0; ii < PARA_FACTOR; ii++) {
#pragma HLS unroll
            int iii = i * PARA_FACTOR + ii;
            float3 velocity;       compute_velocity(density[ii], momentum[ii], velocity);
            float speed_sqd = compute_speed_sqd(velocity);
            float pressure = compute_pressure(density[ii], density_energy[ii], speed_sqd);
            float speed_of_sound = compute_speed_of_sound(density[ii], pressure);
        
            result[iii] = float(0.5f) / (sqrt(areas[iii]) * (sqrt(speed_sqd) + speed_of_sound));
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
#pragma HLS array_partition variable=result_inner cyclic factor=16
    float variables_inner   [TILE_ROWS * NVAR];
#pragma HLS array_partition variable=variables_inner cyclic factor=80
    float areas_inner       [TILE_ROWS];
#pragma HLS array_partition variable=areas_inner cyclic factor=16

    for (int k = 0; k < SIZE / TILE_ROWS; k++){
        memcpy(variables_inner, variables + k * TILE_ROWS * NVAR, sizeof(float) * TILE_ROWS * NVAR);
        memcpy(areas_inner, areas + k * TILE_ROWS, sizeof(float) * TILE_ROWS);

        cfd_step_factor(result_inner, variables_inner, areas_inner);

        memcpy(result + k * TILE_ROWS, result_inner, sizeof(float) * TILE_ROWS);
    }  

    return;
}

}

