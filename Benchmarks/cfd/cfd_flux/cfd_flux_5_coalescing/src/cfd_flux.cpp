#include"cfd_flux.h"
#include"../../../../common/mc.h"

extern "C" {

inline void compute_velocity(float& density, float3& momentum, float3& velocity)
{
    velocity.x = (float)momentum.x / (float)density;
    velocity.y = (float)momentum.y / (float)density;
    velocity.z = (float)momentum.z / (float)density;
}

inline float compute_speed_sqd(float3& velocity)
{
    return (float)velocity.x*(float)velocity.x + (float)velocity.y*(float)velocity.y + (float)velocity.z*(float)velocity.z;
}

inline float compute_pressure(float& density, float& density_energy, float& speed_sqd)
{
    return (float(GAMMA) - float(1.0f))*(float)((float)density_energy - float(0.5f)*(float)((float)density*(float)speed_sqd));
}

inline float compute_speed_of_sound(float& density, float& pressure)
{
    return (float)sqrtf((float)(float(GAMMA)*(float)pressure / (float)density));
}


void cfd_flux(float result[TILE_ROWS *              NVAR], float elements_surrounding_elements [TILE_ROWS * NNB], float normals [TILE_ROWS * NNB * NDIM], float variables [SIZE *              NVAR], float fc_momentum_x [SIZE *       NDIM], float fc_momentum_y [SIZE *       NDIM], float fc_momentum_z [SIZE *       NDIM], float fc_density_energy [SIZE *       NDIM], int k)
{


    const float smoothing_coefficient = float(0.2f);

    float ff_variable[NVAR] = {1.399999976158142090, 1.680000066757202148, 0.000000000000000000, 0.000000000000000000, 3.507999897003173828};

    float3 ff_fc_momentum_x, ff_fc_momentum_y,ff_fc_momentum_z, ff_fc_density_energy;

    ff_fc_momentum_x.x = 3.016000270843505859; ff_fc_momentum_x.y = 0.000000000000000000; ff_fc_momentum_x.z = 0.000000000000000000; 
    ff_fc_momentum_y.x = 0.000000000000000000; ff_fc_momentum_y.y = 1.000000000000000000; ff_fc_momentum_y.z = 0.000000000000000000; 
    ff_fc_momentum_z.x = 0.000000000000000000; ff_fc_momentum_z.y = 0.000000000000000000; ff_fc_momentum_z.z = 1.000000000000000000; 
    ff_fc_density_energy.x = 5.409600257873535156; ff_fc_density_energy.y = 0.000000000000000000; ff_fc_density_energy.z = 0.000000000000000000;


    for(int i = 0; i < TILE_ROWS / PARA_FACTOR; i++)
    {
#pragma HLS pipeline II=1
        for (int ii = 0; ii < PARA_FACTOR; ii++) {
#pragma HLS unroll
            int iii = i * PARA_FACTOR + ii;
        int j, nb;
        float3 normal; float normal_len;
        float factor;

        float density_i = variables[NVAR*iii + VAR_DENSITY];
        float3 momentum_i;
        momentum_i.x = variables[NVAR*iii + (VAR_MOMENTUM+0)];
        momentum_i.y = variables[NVAR*iii + (VAR_MOMENTUM+1)];
        momentum_i.z = variables[NVAR*iii + (VAR_MOMENTUM+2)];
        float density_energy_i = variables[NVAR*iii + VAR_DENSITY_ENERGY];

        float3 velocity_i;                           compute_velocity(density_i, momentum_i, velocity_i);
        float speed_sqd_i                          = compute_speed_sqd(velocity_i);
        float speed_i                              = sqrtf((float)speed_sqd_i);
        float pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
        float speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
        float3 fc_i_momentum_x, fc_i_momentum_y, fc_i_momentum_z;
        float3 fc_i_density_energy;

        fc_i_momentum_x.x = fc_momentum_x[iii*NDIM + 0];
        fc_i_momentum_x.y = fc_momentum_x[iii*NDIM + 1];
        fc_i_momentum_x.z = fc_momentum_x[iii*NDIM + 2];

        fc_i_momentum_y.x = fc_momentum_y[iii*NDIM + 0];
        fc_i_momentum_y.y = fc_momentum_y[iii*NDIM + 1];
        fc_i_momentum_y.z = fc_momentum_y[iii*NDIM + 2];

        fc_i_momentum_z.x = fc_momentum_z[iii*NDIM + 0];
        fc_i_momentum_z.y = fc_momentum_z[iii*NDIM + 1];
        fc_i_momentum_z.z = fc_momentum_z[iii*NDIM + 2];

        fc_i_density_energy.x = fc_density_energy[iii*NDIM + 0];
        fc_i_density_energy.y = fc_density_energy[iii*NDIM + 1];
        fc_i_density_energy.z = fc_density_energy[iii*NDIM + 2];

        float flux_i_density = float(0.0f);
        float3 flux_i_momentum;
        flux_i_momentum.x = float(0.0f);
        flux_i_momentum.y = float(0.0f);
        flux_i_momentum.z = float(0.0f);
        float flux_i_density_energy = float(0.0f);

        float3 velocity_nb;
        float density_nb, density_energy_nb;
        float3 momentum_nb;
        float3 fc_nb_momentum_x, fc_nb_momentum_y, fc_nb_momentum_z;
        float3 fc_nb_density_energy;
        float speed_sqd_nb, speed_of_sound_nb, pressure_nb;
        for(j = 0; j < 4; j++)
        {
#pragma HLS unroll
            nb = elements_surrounding_elements[iii*NNB + j];
            normal.x = normals[(iii*NNB + j)*NDIM + 0];
            normal.y = normals[(iii*NNB + j)*NDIM + 1];
            normal.z = normals[(iii*NNB + j)*NDIM + 2];
           normal_len = sqrtf((float)((float)((float)normal.x*(float)normal.x) + (float)((float)normal.y*(float)normal.y) + (float)((float)normal.z*(float)normal.z)));

            if(nb >= 0)     // a legitimate neighbor
            {
                density_nb =        variables[nb*NVAR + VAR_DENSITY];
                momentum_nb.x =     variables[nb*NVAR + (VAR_MOMENTUM+0)];
                momentum_nb.y =     variables[nb*NVAR + (VAR_MOMENTUM+1)];
                momentum_nb.z =     variables[nb*NVAR + (VAR_MOMENTUM+2)];
                density_energy_nb = variables[nb*NVAR + VAR_DENSITY_ENERGY];
                                                    compute_velocity(density_nb, momentum_nb, velocity_nb);
                speed_sqd_nb                      = compute_speed_sqd(velocity_nb);
                pressure_nb                       = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
                speed_of_sound_nb                 = compute_speed_of_sound(density_nb, pressure_nb);

                fc_nb_momentum_x.x = fc_momentum_x[nb*NDIM + 0];
                fc_nb_momentum_x.y = fc_momentum_x[nb*NDIM + 1];
                fc_nb_momentum_x.z = fc_momentum_x[nb*NDIM + 2];

                fc_nb_momentum_y.x = fc_momentum_y[nb*NDIM + 0];
                fc_nb_momentum_y.y = fc_momentum_y[nb*NDIM + 1];
                fc_nb_momentum_y.z = fc_momentum_y[nb*NDIM + 2];

                fc_nb_momentum_z.x = fc_momentum_z[nb*NDIM + 0];
                fc_nb_momentum_z.y = fc_momentum_z[nb*NDIM + 1];
                fc_nb_momentum_z.z = fc_momentum_z[nb*NDIM + 2];

                fc_nb_density_energy.x = fc_density_energy[nb*NDIM + 0];
                fc_nb_density_energy.y = fc_density_energy[nb*NDIM + 1];
                fc_nb_density_energy.z = fc_density_energy[nb*NDIM + 2];

                // artificial viscosity
                factor = -(float)normal_len*(float)((float)smoothing_coefficient*(float)(0.5*(float)(speed_i + sqrtf((float)speed_sqd_nb) + (float)speed_of_sound_i + (float)speed_of_sound_nb)));
                flux_i_density += (float)((float)factor*(float)((float)density_i-(float)density_nb));
                flux_i_density_energy += (float)((float)factor*(float)((float)density_energy_i-(float)density_energy_nb));
                flux_i_momentum.x += (float)factor*(float)((float)momentum_i.x-(float)momentum_nb.x);
                flux_i_momentum.y += (float)factor*((float)momentum_i.y-(float)momentum_nb.y);
                flux_i_momentum.z += (float)factor*((float)momentum_i.z-(float)momentum_nb.z);

                // accumulate cell-centered fluxes
                factor = (float)float(0.5f)*(float)normal.x;
                flux_i_density += (float)factor*(float)((float)momentum_nb.x+(float)momentum_i.x);
                flux_i_density_energy += (float)factor*((float)fc_nb_density_energy.x+(float)fc_i_density_energy.x);
                flux_i_momentum.x += (float)factor*(float)((float)fc_nb_momentum_x.x+(float)fc_i_momentum_x.x);
                flux_i_momentum.y += (float)factor*(float)((float)fc_nb_momentum_y.x+(float)fc_i_momentum_y.x);
                flux_i_momentum.z += (float)factor*(float)((float)fc_nb_momentum_z.x+(float)fc_i_momentum_z.x);

                factor = (float)float(0.5f)*(float)normal.y;
                flux_i_density += (float)factor*(float)((float)momentum_nb.y+(float)momentum_i.y);
                flux_i_density_energy += (float)factor*(float)((float)fc_nb_density_energy.y+(float)fc_i_density_energy.y);
                flux_i_momentum.x += (float)factor*(float)((float)fc_nb_momentum_x.y+(float)fc_i_momentum_x.y);
                flux_i_momentum.y += (float)factor*(float)((float)fc_nb_momentum_y.y+(float)fc_i_momentum_y.y);
                flux_i_momentum.z += (float)factor*(float)((float)fc_nb_momentum_z.y+(float)fc_i_momentum_z.y);

                factor = (float)float(0.5f)*(float)normal.z;
                flux_i_density += (float)factor*(float)((float)momentum_nb.z+(float)momentum_i.z);
                flux_i_density_energy += (float)factor*(float)((float)fc_nb_density_energy.z+(float)fc_i_density_energy.z);
                flux_i_momentum.x += (float)factor*(float)((float)fc_nb_momentum_x.z+(float)fc_i_momentum_x.z);
                flux_i_momentum.y += (float)factor*(float)((float)fc_nb_momentum_y.z+(float)fc_i_momentum_y.z);
                flux_i_momentum.z += (float)factor*(float)((float)fc_nb_momentum_z.z+(float)fc_i_momentum_z.z);
            }
            else if(nb == -1)   // a wing boundary
            {
                flux_i_momentum.x += (float)normal.x*(float)pressure_i;
                flux_i_momentum.y += (float)normal.y*(float)pressure_i;
                flux_i_momentum.z += (float)normal.z*(float)pressure_i;
            }
            else if(nb == -2) // a far field boundary
            {
                factor = float(0.5f)*(float)normal.x;
                flux_i_density += (float)factor*(float)((float)ff_variable[VAR_MOMENTUM+0]+(float)momentum_i.x);
                flux_i_density_energy += (float)factor*((float)ff_fc_density_energy.x+(float)fc_i_density_energy.x);
                flux_i_momentum.x += (float)factor*(float)((float)ff_fc_momentum_x.x + (float)fc_i_momentum_x.x);
                flux_i_momentum.y += (float)factor*(float)((float)ff_fc_momentum_y.x + (float)fc_i_momentum_y.x);
                flux_i_momentum.z += (float)factor*(float)((float)ff_fc_momentum_z.x + (float)fc_i_momentum_z.x);

                factor = float(0.5f)*(float)normal.y;
                flux_i_density += (float)factor*(float)((float)ff_variable[VAR_MOMENTUM+1]+(float)momentum_i.y);
                flux_i_density_energy += (float)factor*(float)((float)ff_fc_density_energy.y+(float)fc_i_density_energy.y);
                flux_i_momentum.x += (float)factor*(float)((float)ff_fc_momentum_x.y + (float)fc_i_momentum_x.y);
                flux_i_momentum.y += (float)factor*(float)((float)ff_fc_momentum_y.y + (float)fc_i_momentum_y.y);
                flux_i_momentum.z += (float)factor*(float)((float)ff_fc_momentum_z.y + (float)fc_i_momentum_z.y);

                factor = float(0.5f)*(float)normal.z;
                flux_i_density += (float)factor*(float)((float)ff_variable[VAR_MOMENTUM+2]+(float)momentum_i.z);
                flux_i_density_energy += (float)factor*((float)ff_fc_density_energy.z+(float)fc_i_density_energy.z);
                flux_i_momentum.x += (float)factor*(float)((float)ff_fc_momentum_x.z + (float)fc_i_momentum_x.z);
                flux_i_momentum.y += (float)factor*(float)((float)ff_fc_momentum_y.z + (float)fc_i_momentum_y.z);
                flux_i_momentum.z += (float)factor*(float)((float)ff_fc_momentum_z.z + (float)fc_i_momentum_z.z);

            }
        }

        result[iii*NVAR + VAR_DENSITY] = flux_i_density;
        result[iii*NVAR + (VAR_MOMENTUM+0)] = flux_i_momentum.x;
        result[iii*NVAR + (VAR_MOMENTUM+1)] = flux_i_momentum.y;
        result[iii*NVAR + (VAR_MOMENTUM+2)] = flux_i_momentum.z;
        result[iii*NVAR + VAR_DENSITY_ENERGY] = flux_i_density_energy;
    }
    }

    return;
}

void buffer_load(int flag, int k, float elements_surrounding_elements_inner[TILE_ROWS  * NNB], class ap_uint<LARGE_BUS> * elements_surrounding_elements, float normals_inner[TILE_ROWS  * NNB * NDIM ], class ap_uint<LARGE_BUS> * normals)
{
#pragma HLS inline off
    if (flag) {
		memcpy_wide_bus_read_float(elements_surrounding_elements_inner, elements_surrounding_elements + k * TILE_ROWS  * NNB/ (LARGE_BUS / 32),0, sizeof(float) * TILE_ROWS  * NNB);
		memcpy_wide_bus_read_float(normals_inner, normals + k * TILE_ROWS  * NNB * NDIM / (LARGE_BUS / 32) ,0, sizeof(float) * TILE_ROWS  * NNB * NDIM );
		}
    return;
}

void buffer_compute(int flag, float result[TILE_ROWS  * NVAR], float elements_surrounding_elements[TILE_ROWS  * NNB], float normals[TILE_ROWS  * NNB * NDIM], float variables [SIZE *              NVAR], float fc_momentum_x [SIZE *       NDIM], float fc_momentum_y [SIZE *       NDIM], float fc_momentum_z [SIZE *       NDIM], float fc_density_energy [SIZE *       NDIM], int k)
{
#pragma HLS inline off
 if (flag)   cfd_flux(result, elements_surrounding_elements, normals, variables, fc_momentum_x, fc_momentum_y, fc_momentum_z, fc_density_energy, k);
    return;
}

void buffer_store(int flag, int k, class ap_uint<LARGE_BUS> * result, float result_inner[TILE_ROWS  *              NVAR])
{
#pragma HLS inline off
    if (flag) memcpy_wide_bus_write_float(result + k * TILE_ROWS  * NVAR / (LARGE_BUS / 32), result_inner,0, sizeof(float) * TILE_ROWS  *              NVAR);
    return;
}

void workload(class ap_uint<LARGE_BUS> * result, class ap_uint<LARGE_BUS> * elements_surrounding_elements , class ap_uint<LARGE_BUS> * normals , class ap_uint<LARGE_BUS> * variables, class ap_uint<LARGE_BUS> * fc_momentum_x, class ap_uint<LARGE_BUS> * fc_momentum_y, class ap_uint<LARGE_BUS> * fc_momentum_z, class ap_uint<LARGE_BUS> * fc_density_energy)
{


    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=result


    #pragma HLS INTERFACE m_axi port=elements_surrounding_elements offset=slave bundle=elements_surrounding_elements
    #pragma HLS INTERFACE m_axi port=normals offset=slave bundle=normals
    #pragma HLS INTERFACE m_axi port=variables offset=slave bundle=variables
    #pragma HLS INTERFACE m_axi port=fc_momentum_x offset=slave bundle=fc_momentum_x
    #pragma HLS INTERFACE m_axi port=fc_momentum_y offset=slave bundle=fc_momentum_y
    #pragma HLS INTERFACE m_axi port=fc_momentum_z offset=slave bundle=fc_momentum_z
    #pragma HLS INTERFACE m_axi port=fc_density_energy offset=slave bundle=fc_density_energy
    
    #pragma HLS INTERFACE s_axilite port=result bundle=control

    #pragma HLS INTERFACE s_axilite port=elements_surrounding_elements bundle=control
    #pragma HLS INTERFACE s_axilite port=normals bundle=control
    #pragma HLS INTERFACE s_axilite port=variables bundle=control
    #pragma HLS INTERFACE s_axilite port=fc_momentum_x bundle=control
    #pragma HLS INTERFACE s_axilite port=fc_momentum_y bundle=control
    #pragma HLS INTERFACE s_axilite port=fc_momentum_z bundle=control
    #pragma HLS INTERFACE s_axilite port=fc_density_energy bundle=control

    #pragma HLS INTERFACE s_axilite port=return bundle=control

    float result_inner_0                          [TILE_ROWS  *              NVAR ];
    float elements_surrounding_elements_inner_0   [TILE_ROWS  * NNB               ];
    float normals_inner_0                         [TILE_ROWS  * NNB * NDIM        ];

    float result_inner_1                          [TILE_ROWS  *              NVAR ];
    float elements_surrounding_elements_inner_1   [TILE_ROWS  * NNB               ];
    float normals_inner_1                         [TILE_ROWS  * NNB * NDIM        ];

    float result_inner_2                          [TILE_ROWS  *              NVAR ];
    float elements_surrounding_elements_inner_2   [TILE_ROWS  * NNB               ];
    float normals_inner_2                         [TILE_ROWS  * NNB * NDIM        ];

    float variables_inner                       [SIZE       *       NVAR        ];
    float fc_momentum_x_inner                   [SIZE       *       NDIM        ];
    float fc_momentum_y_inner                   [SIZE       *       NDIM        ];
    float fc_momentum_z_inner                   [SIZE       *       NDIM        ];
    float fc_density_energy_inner               [SIZE       *       NDIM        ];

    memcpy_wide_bus_read_float(variables_inner, variables,0,                  sizeof(float) * SIZE       *              NVAR );
    memcpy_wide_bus_read_float(fc_momentum_x_inner, fc_momentum_x,0,          sizeof(float) * SIZE       *       NDIM        );
    memcpy_wide_bus_read_float(fc_momentum_y_inner, fc_momentum_y,0,          sizeof(float) * SIZE       *       NDIM        );
    memcpy_wide_bus_read_float(fc_momentum_z_inner, fc_momentum_z,0,          sizeof(float) * SIZE       *       NDIM        );
    memcpy_wide_bus_read_float(fc_density_energy_inner, fc_density_energy,0,  sizeof(float) * SIZE       *       NDIM        );

    for (int k = 0; k < SIZE / TILE_ROWS + 2; k++) {

        int load_flag = k >= 0 && k < (SIZE / TILE_ROWS);
        int compute_flag = k >= 1 && k < (SIZE / TILE_ROWS) + 1;
        int store_flag = k >= 2 && k < (SIZE / TILE_ROWS) + 2;
        

        if (k % 3 == 0) {
            buffer_load(load_flag, k, elements_surrounding_elements_inner_0, elements_surrounding_elements, normals_inner_0, normals);
            buffer_compute(compute_flag, result_inner_2, elements_surrounding_elements_inner_2, normals_inner_2, variables_inner, fc_momentum_x_inner, fc_momentum_y_inner, fc_momentum_z_inner, fc_density_energy_inner, k);
            buffer_store(store_flag, k - 2, result, result_inner_1);
        }
        
        else if (k % 3 == 1) {
            buffer_load(load_flag, k, elements_surrounding_elements_inner_1, elements_surrounding_elements, normals_inner_1, normals);
            buffer_compute(compute_flag, result_inner_0, elements_surrounding_elements_inner_0, normals_inner_0, variables_inner, fc_momentum_x_inner, fc_momentum_y_inner, fc_momentum_z_inner, fc_density_energy_inner, k);
            buffer_store(store_flag, k - 2, result, result_inner_2);
        }
        
        else {
            buffer_load(load_flag, k, elements_surrounding_elements_inner_2, elements_surrounding_elements, normals_inner_2, normals);
            buffer_compute(compute_flag, result_inner_1, elements_surrounding_elements_inner_1, normals_inner_1, variables_inner, fc_momentum_x_inner, fc_momentum_y_inner, fc_momentum_z_inner, fc_density_energy_inner, k);
            buffer_store(store_flag, k - 2, result, result_inner_0);
        }
    }
	return;

}

}

