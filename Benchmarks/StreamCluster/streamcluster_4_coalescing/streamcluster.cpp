#include "streamcluster.h"
#include "../../../common/mc.h"
#define BUF_SIZE 256

extern "C" {

void load(int flag, int i, class ap_uint<LARGE_BUS>* coord, class ap_uint<LARGE_BUS>* weight, class ap_uint<LARGE_BUS>* cost, class ap_uint<LARGE_BUS>* assign,
			float* buffer_coord, float* buffer_weight, float* buffer_cost, int* buffer_assign){
	if(flag){
		memcpy_wide_bus_read_float(buffer_coord, (class ap_uint<LARGE_BUS>*)(coord + i * DIM / (LARGE_BUS / 32)), 0, BUF_SIZE * DIM * sizeof(float));
		memcpy_wide_bus_read_float(buffer_weight, (class ap_uint<LARGE_BUS>*)(weight + i / (LARGE_BUS / 32)), 0, BUF_SIZE * sizeof(float));
		memcpy_wide_bus_read_float(buffer_cost, (class ap_uint<LARGE_BUS>*)(cost + i / (LARGE_BUS / 32)), 0, BUF_SIZE * sizeof(float));
		memcpy_wide_bus_read_int(buffer_assign, (class ap_uint<LARGE_BUS>*)(assign + i / (LARGE_BUS / 32)), 0, BUF_SIZE * sizeof(int));
	}
}

//Input of compute should be the buffer 
void compute(int flag, int num, int k, float* coord, float* weight, float* target, float* cost, int * assign, 
		int* center_table, char* switch_membership, float* cost_of_opening_x, float* work_mem, float* x_cost){
	if(flag){
		pre:for(int i = 0; i < BUF_SIZE; i++){
			#pragma pipeline II=1
			#pragma HLS DEPENDENCE variable=x_cost inter false 
			float sum = 0;
			pre_inner:for(int j = 0; j < DIM; j++){
				#pragma HLS unroll
				float a = coord[i * DIM + j] - target[j];
				sum += a * a;
			}
			x_cost[i] = sum * weight[i];
		}

		after:for(int i = 0; i < BUF_SIZE; i++){
			//#pragma HLS pipeline II=6
			float current_cost = x_cost[i] - cost[i];

			if(current_cost < 0){
				switch_membership[k + i] = 1;
				cost_of_opening_x[0] += current_cost;
			} else{
				int local_center_index = center_table[assign[i]];
				work_mem[local_center_index] -= current_cost;
			}
		}
	}
}

void store(int num, int numcenter, class ap_uint<LARGE_BUS>* work_mem, class ap_uint<LARGE_BUS>* switch_membership, float* cost_of_opening_x,
			float* buffer_work_mem, char* buffer_switch_membership, float* buffer_cost_of_opening_x){
	memcpy_wide_bus_write_float((class ap_uint<LARGE_BUS>*) work_mem, buffer_work_mem, 0, BATCH_SIZE * sizeof(float));
	memcpy_wide_bus_write_char((class ap_uint<LARGE_BUS>*)switch_membership, buffer_switch_membership, 0, BATCH_SIZE * sizeof(char));
	// memcpy(work_mem, buffer_work_mem, BATCH_SIZE * sizeof(float));
	// memcpy(switch_membership, buffer_switch_membership, BATCH_SIZE * sizeof(char));
	cost_of_opening_x[0] = buffer_cost_of_opening_x[0];
}

void workload(    
    class ap_uint<LARGE_BUS> * coord,    //float                  
    class ap_uint<LARGE_BUS>* weight,    //float           
    class ap_uint<LARGE_BUS>* cost, 	//float
    class ap_uint<LARGE_BUS>* target,	//float
    class ap_uint<LARGE_BUS>* assign,    //int
    class ap_uint<LARGE_BUS>* center_table,//int
    class ap_uint<LARGE_BUS>* switch_membership, // char
    class ap_uint<LARGE_BUS>* work_mem, // float
    int num,
    float* cost_of_opening_x,
    int numcenter            
)
{
    #pragma HLS INTERFACE m_axi port=coord offset=slave bundle=gmemf
    #pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmemf
    #pragma HLS INTERFACE m_axi port=cost offset=slave bundle=gmemf
    #pragma HLS INTERFACE m_axi port=target offset=slave bundle=gmemf
    #pragma HLS INTERFACE m_axi port=assign offset=slave bundle=gmemi
    #pragma HLS INTERFACE m_axi port=center_table offset=slave bundle=gmemi
    #pragma HLS INTERFACE m_axi port=switch_membership offset=slave bundle=gmemc
    #pragma HLS INTERFACE m_axi port=work_mem offset=slave bundle=gmemf
    #pragma HLS INTERFACE m_axi port=cost_of_opening_x offset=slave bundle=gmemf

    #pragma HLS INTERFACE s_axilite port=coord bundle=control
    #pragma HLS INTERFACE s_axilite port=weight bundle=control
    #pragma HLS INTERFACE s_axilite port=cost bundle=control
    #pragma HLS INTERFACE s_axilite port=target bundle=control
    #pragma HLS INTERFACE s_axilite port=assign bundle=control
    #pragma HLS INTERFACE s_axilite port=center_table bundle=control
    #pragma HLS INTERFACE s_axilite port=switch_membership bundle=control
    #pragma HLS INTERFACE s_axilite port=work_mem bundle=control
    #pragma HLS INTERFACE s_axilite port=num bundle=control
    #pragma HLS INTERFACE s_axilite port=cost_of_opening_x bundle=control
    #pragma HLS INTERFACE s_axilite port=numcenter bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

	float buffer_target[DIM];
	#pragma HLS array_partition variable=buffer_target complete

	int buffer_center_table[BATCH_SIZE];
	#pragma HLS array_partition variable=buffer_center_table cyclic factor=8
	float buffer_work_mem[BATCH_SIZE];
	#pragma HLS array_partition variable=buffer_work_mem cyclic factor=8
	char buffer_switch_membership[BATCH_SIZE];
	#pragma HLS array_partition variable=buffer_switch_membership cyclic factor=8
	float buffer_cost_of_opening_x[1];
	buffer_cost_of_opening_x[0] = cost_of_opening_x[0];

	float x_cost[BUF_SIZE];
	#pragma HLS array_partition variable=x_cost complete

	int buffer_assign_1[BUF_SIZE];
	#pragma HLS array_partition variable=buffer_assign_1 cyclic factor=8
	float buffer_cost_1[BUF_SIZE];
	#pragma HLS array_partition variable=buffer_cost_1 cyclic factor=8
	float buffer_weight_1[BUF_SIZE];
	#pragma HLS array_partition variable=buffer_weight_1 cyclic factor=8
	float buffer_coord_1[BUF_SIZE * DIM];
	#pragma HLS array_partition variable=buffer_coord_1 cyclic factor=200 //BUF_SIZE

	int buffer_assign_2[BUF_SIZE];
	#pragma HLS array_partition variable=buffer_assign_2 cyclic factor=8
	float buffer_cost_2[BUF_SIZE];
	#pragma HLS array_partition variable=buffer_cost_2 cyclic factor=8
	float buffer_weight_2[BUF_SIZE];
	#pragma HLS array_partition variable=buffer_weight_2 cyclic factor=8
	float buffer_coord_2[BUF_SIZE * DIM];
	#pragma HLS array_partition variable=buffer_coord_2 cyclic factor=200 //BUF_SIZE
	// float x_cost_2[BUF_SIZE];
	// #pragma HLS array_partition variable=x_cost_2 complete

	memcpy_wide_bus_read_float(buffer_work_mem, (class ap_uint<LARGE_BUS>*)work_mem, 0, BATCH_SIZE * sizeof(float));
	memcpy_wide_bus_read_char(buffer_switch_membership, (class ap_uint<LARGE_BUS>*)switch_membership, 0, BATCH_SIZE * sizeof(char));

	// memcpy(buffer_work_mem, work_mem, BATCH_SIZE * sizeof(float));
	// memcpy(buffer_switch_membership, switch_membership, BATCH_SIZE * sizeof(float));
	
	memcpy_wide_bus_read_float(buffer_target, (class ap_uint<LARGE_BUS>*)target, 0, DIM * sizeof(float));
	memcpy_wide_bus_read_int(buffer_center_table, (class ap_uint<LARGE_BUS>*)center_table, 0, BATCH_SIZE * sizeof(int));
	// memcpy(buffer_target, target, DIM * sizeof(float));
	// memcpy(buffer_center_table, center_table, BATCH_SIZE * sizeof(int));

	process:for(int i = 0; i < BATCH_SIZE + BUF_SIZE; i += BUF_SIZE){
		int load_flag = i >= 0 && i < BATCH_SIZE;
		int compute_flag = i >= BUF_SIZE && i <= BATCH_SIZE + BUF_SIZE;

		if((i/BUF_SIZE) % 2 == 0){
			load(load_flag, i, coord, weight, cost, assign, buffer_coord_1, buffer_weight_1, buffer_cost_1, buffer_assign_1);
			compute(compute_flag, num, i - BUF_SIZE, buffer_coord_2, buffer_weight_2, buffer_target, buffer_cost_2, buffer_assign_2, 
					buffer_center_table, buffer_switch_membership, buffer_cost_of_opening_x, buffer_work_mem, x_cost);
		} else{
			load(load_flag, i, coord, weight, cost, assign, buffer_coord_2, buffer_weight_2, buffer_cost_2, buffer_assign_2);
			compute(compute_flag, num, i - BUF_SIZE, buffer_coord_1, buffer_weight_1, buffer_target, buffer_cost_1, buffer_assign_1, 
					buffer_center_table, buffer_switch_membership, buffer_cost_of_opening_x, buffer_work_mem, x_cost);
		}

	}

	store(num, numcenter, work_mem, switch_membership, cost_of_opening_x,
		buffer_work_mem, buffer_switch_membership, buffer_cost_of_opening_x);

	return;
}
}
