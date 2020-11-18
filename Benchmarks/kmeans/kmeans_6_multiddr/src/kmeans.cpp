#include "kmeans.h"

extern "C"{
void load_local_cluster(float local_clusters[NCLUSTERS * NFEATURES], MARS_WIDE_BUS_TYPE *clusters)
		//float clusters[NCLUSTERS * NFEATURES])
{
	int num_transfers = (NCLUSTERS*NFEATURES) / WIDTH_FACTOR;
	int remainder = (NCLUSTERS*NFEATURES) % WIDTH_FACTOR;
	for (int i(0); i<num_transfers; ++i){
#pragma HLS PIPELINE II=1
		memcpy_wide_bus_read_float(local_clusters+(i*WIDTH_FACTOR), (class ap_uint<LARGE_BUS> *)(clusters+i), 0, 64);
	}
	memcpy_wide_bus_read_float(local_clusters+(num_transfers*WIDTH_FACTOR), (class ap_uint<LARGE_BUS> *)(clusters+num_transfers), 0, 4*remainder);
}

void load_local_feature(int flag, float local_feature[TILE_SIZE * NFEATURES], MARS_WIDE_BUS_TYPE *feature, int tile_idx)
{
	if (flag){
		int num_transfers = (TILE_SIZE * NFEATURES) / WIDTH_FACTOR;
		int start_idx = tile_idx * ((TILE_SIZE * NFEATURES) / WIDTH_FACTOR);
		for (int i(0); i<num_transfers; ++i){
			#pragma HLS PIPELINE II=1
			memcpy_wide_bus_read_float(local_feature+(i*WIDTH_FACTOR), (class ap_uint<LARGE_BUS> *)(feature+start_idx+i), 0, 64);
		}
	}
}

void compute_local_membership(int flag, float local_feature[TILE_SIZE * NFEATURES],
		float local_clusters[NCLUSTERS * NFEATURES],int local_membership[TILE_SIZE])
{
	if (flag){
    for (int i = 0; i < TILE_SIZE; i++) {
#pragma HLS PIPELINE II=1
    	float min_dist = FLT_MAX;
        int index = 0;

        /* find the cluster center id with min distance to pt */
        MIN: for (int j = 0; j < NCLUSTERS; j++) {
#pragma HLS UNROLL
        	float dist = 0.0;

            DIST: for (int k = 0; k < NFEATURES; k++) {
#pragma HLS UNROLL
                float diff = local_feature[NFEATURES * i + k] - local_clusters[NFEATURES * j + k];
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                index = j;
            }
        }

        /* assign the membership to object i */
        local_membership[i] = index;
    }
	}
}

void store_local_membership(int flag, int local_membership[TILE_SIZE], MARS_WIDE_BUS_TYPE *membership, int tile_idx)
{
	if (flag){
		int num_transfers = TILE_SIZE / WIDTH_FACTOR;
		int start_idx = tile_idx * (TILE_SIZE / WIDTH_FACTOR);
		for (int i(0); i<num_transfers; ++i){
#pragma HLS PIPELINE II=1
			memcpy_wide_bus_write_int((class ap_uint<LARGE_BUS> *)(membership+(start_idx+i)), local_membership+(i*WIDTH_FACTOR), 0, 64);
		}
	}
}

void workload(MARS_WIDE_BUS_TYPE  *feature, /* [npoints][nfeatures] */
			  MARS_WIDE_BUS_TYPE  *clusters, /* [n_clusters][n_features] */
			  MARS_WIDE_BUS_TYPE  *membership)
{
#pragma HLS INTERFACE m_axi port=feature offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=clusters offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=membership offset=slave bundle=gmem2

#pragma HLS INTERFACE s_axilite port=feature bundle=control
#pragma HLS INTERFACE s_axilite port=membership bundle=control
#pragma HLS INTERFACE s_axilite port=clusters bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int local_membership_0[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=local_membership_0 cyclic factor=WIDTH_FACTOR
    float local_feature_0[TILE_SIZE * NFEATURES];
#pragma HLS ARRAY_PARTITION variable=local_feature_0 cyclic factor=34 //NFEATURES

    int local_membership_1[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=local_membership_1 cyclic factor=WIDTH_FACTOR
    float local_feature_1[TILE_SIZE * NFEATURES];
#pragma HLS ARRAY_PARTITION variable=local_feature_1 cyclic factor=34 //NFEATURES

    int local_membership_2[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=local_membership_2 cyclic factor=WIDTH_FACTOR
    float local_feature_2[TILE_SIZE * NFEATURES];
#pragma HLS ARRAY_PARTITION variable=local_feature_2 cyclic factor=34 //NFEATURES

    float local_clusters[NCLUSTERS * NFEATURES];
#pragma HLS ARRAY_PARTITION variable=local_clusters complete

    load_local_cluster(local_clusters, clusters);

    for (int i=0; i<NUM_TILES+2; ++i)
    {
        int load_flag = (i >= 0) && (i < NUM_TILES);
        int compute_flag = (i >= 1) && (i < NUM_TILES+1);
        int store_flag = (i >= 2) && (i < NUM_TILES+2);

        if (i % 3 == 0){
	    	load_local_feature(load_flag, local_feature_0, feature, i);
	    	compute_local_membership(compute_flag, local_feature_2, local_clusters, local_membership_2);
	        store_local_membership(store_flag, local_membership_1, membership, i-2);
        }
        else if (i % 3 == 1){
	    	load_local_feature(load_flag, local_feature_1, feature, i);
	    	compute_local_membership(compute_flag, local_feature_0, local_clusters, local_membership_0);
	        store_local_membership(store_flag, local_membership_2, membership, i-2);
        }
        else{
	    	load_local_feature(load_flag, local_feature_2, feature, i);
	    	compute_local_membership(compute_flag, local_feature_1, local_clusters, local_membership_1);
	        store_local_membership(store_flag, local_membership_0, membership, i-2);
        }


    }
}

}