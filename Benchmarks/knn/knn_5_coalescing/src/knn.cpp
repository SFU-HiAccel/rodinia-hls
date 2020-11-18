#include "knn.h"

void load (int flag, int load_idx, INTERFACE_WIDTH* searchSpace, INTERFACE_WIDTH* local_searchSpace)
{
#pragma HLS INLINE OFF
    if (flag){
        int start_idx = load_idx * NUM_PT_IN_BUFFER * NUM_FEATURE / WIDTH_FACTOR;
        LOAD_TILE: for (int i(0); i < NUM_PT_IN_BUFFER*NUM_FEATURE/WIDTH_FACTOR; ++i){
        #pragma HLS PIPELINE II=1
            local_searchSpace[i] = searchSpace[start_idx+i];
        }
    }
}

void compute (int flag, float* local_inputQuery, INTERFACE_WIDTH* local_searchSpace, INTERFACE_WIDTH* local_distance)
{
#pragma HLS INLINE OFF
    if (flag){
        COMPUTE_TILE_INPUT_MAJOR: 
        for (int i = 0; i < NUM_PT_IN_BUFFER*NUM_FEATURE/WIDTH_FACTOR; i+=UNROLL_FACTOR){
        #pragma HLS PIPELINE II=1
            for (int n = 0; n < UNROLL_FACTOR; ++n){
            #pragma HLS UNROLL
                for (int j = 0; j < WIDTH_FACTOR; j+=NUM_FEATURE){
                    float feature_delta = 0.0;
                    float sum = 0.0;
                    for (int k = 0; k < NUM_FEATURE; ++k){
                        unsigned int range_idx = (j+k) * 32;
                        uint32_t feature_item = local_searchSpace[i+n].range(range_idx+31, range_idx);
                        float feature_item_value = *((float*)(&feature_item));
                        feature_delta = feature_item_value - local_inputQuery[k];
                        sum += feature_delta * feature_delta;
                    }
                    unsigned int range_idx = (((i+n)%2)*WIDTH_FACTOR/2+ j/2) * 32;
                    local_distance[(i+n)/2].range(range_idx+31, range_idx) = *((uint32_t *)(&sum));
                }
            }
        }
    }
}

void store (int flag, int store_idx, INTERFACE_WIDTH* local_distance, INTERFACE_WIDTH* distance)
{
#pragma HLS INLINE OFF
    if (flag){
        int start_idx = store_idx * NUM_PT_IN_BUFFER / WIDTH_FACTOR;
        STORE_TILE: for (int i(0); i < NUM_PT_IN_BUFFER/WIDTH_FACTOR; ++i){
        #pragma HLS PIPELINE II=1
            distance[start_idx+i] = local_distance[i];
        }
    }
}

void workload(
	float inputQuery[NUM_FEATURE],
	INTERFACE_WIDTH searchSpace[NUM_PT_IN_SEARCHSPACE*NUM_FEATURE/WIDTH_FACTOR],
    INTERFACE_WIDTH distance[NUM_PT_IN_SEARCHSPACE/WIDTH_FACTOR]
){
    // #pragma HLS INTERFACE m_axi port=inputQuery offset=slave bundle=gmem
    // #pragma HLS INTERFACE s_axilite port=inputQuery bundle=control
    // #pragma HLS INTERFACE m_axi port=searchSpace offset=slave bundle=gmem
    // #pragma HLS INTERFACE s_axilite port=searchSpace bundle=control
    // #pragma HLS INTERFACE m_axi port=distance offset=slave bundle=gmem
    // #pragma HLS INTERFACE s_axilite port=distance bundle=control
    // #pragma HLS INTERFACE s_axilite port=return bundle=control

	float local_inputQuery[NUM_FEATURE];
    #pragma HLS ARRAY_PARTITION variable=local_inputQuery complete
	
    INTERFACE_WIDTH local_searchSpace_0[NUM_PT_IN_BUFFER*NUM_FEATURE/WIDTH_FACTOR];
    #pragma HLS ARRAY_PARTITION variable=local_searchSpace_0 complete
    INTERFACE_WIDTH local_searchSpace_1[NUM_PT_IN_BUFFER*NUM_FEATURE/WIDTH_FACTOR];
    #pragma HLS ARRAY_PARTITION variable=local_searchSpace_1 complete
	
    INTERFACE_WIDTH local_distance_0[NUM_PT_IN_BUFFER/WIDTH_FACTOR];
    INTERFACE_WIDTH local_distance_1[NUM_PT_IN_BUFFER/WIDTH_FACTOR];
    
	LOAD_INPUTQUERY: for (int i(0); i<NUM_FEATURE; ++i){
	#pragma HLS PIPELINE II=1
		local_inputQuery[i] = inputQuery[i];
    }
	
	TILED_PE: for (int tile_idx(0); tile_idx<NUM_TILES+2; ++tile_idx){
		int load_flag = tile_idx >= 0 && tile_idx < NUM_TILES;
		int compute_flag = tile_idx >= 1 && tile_idx < NUM_TILES + 1;
		int store_flag = tile_idx >= 2 && tile_idx < NUM_TILES + 2;

	    if (tile_idx % 2 == 0) {
	    	load(load_flag, tile_idx, searchSpace, local_searchSpace_0);
			compute(compute_flag, local_inputQuery, local_searchSpace_1, local_distance_1);
			store(store_flag, tile_idx-2, local_distance_0, distance);
	    }
	    else {
	    	load(load_flag, tile_idx, searchSpace, local_searchSpace_1);
			compute(compute_flag, local_inputQuery, local_searchSpace_0, local_distance_0);
			store(store_flag, tile_idx-2, local_distance_1, distance);
	    }
	}

	return;
} 