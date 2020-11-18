#include "kmeans.h"

extern "C"{
void load_local_cluster(float local_clusters[NCLUSTERS * NFEATURES], float clusters[NCLUSTERS * NFEATURES])
{
	for (int i(0); i<NCLUSTERS; ++i){
		for (int j(0); j<NFEATURES; ++j){
#pragma HLS PIPELINE II=1
			local_clusters[i*NFEATURES+j] = clusters[i*NFEATURES+j];
		}
	}
}

void load_local_feature(float local_feature[TILE_SIZE * NFEATURES], float feature[NPOINTS * NFEATURES], int tile_idx)
{
	for (int i(0); i<TILE_SIZE; ++i){
		for (int j(0); j<NFEATURES; ++j){
#pragma HLS PIPELINE II=1
			local_feature[i*NFEATURES+j] = feature[(tile_idx*TILE_SIZE+i)*NFEATURES+j];
		}
	}
}

void compute_local_membership(float local_feature[TILE_SIZE * NFEATURES],
		float local_clusters[NCLUSTERS * NFEATURES],int local_membership[TILE_SIZE])
{

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

void store_local_membership(int local_membership[TILE_SIZE], int membership[NPOINTS], int tile_idx)
{
	for (int i(0); i<TILE_SIZE; ++i){
#pragma HLS PIPELINE II=1
		membership[tile_idx*TILE_SIZE+i] = local_membership[i];
	}
}

void workload(float  *feature, /* [npoints][nfeatures] */
              float  *clusters, /* [n_clusters][n_features] */
			  int *membership)
{
#pragma HLS INTERFACE m_axi port=feature offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=membership offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=clusters offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=feature bundle=control
#pragma HLS INTERFACE s_axilite port=membership bundle=control
#pragma HLS INTERFACE s_axilite port=clusters bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int i, ii, j, k, index;

    int local_membership[TILE_SIZE];
    float local_feature[TILE_SIZE * NFEATURES];
#pragma HLS ARRAY_PARTITION variable=local_feature cyclic factor=34 //NFEATURES
    float local_clusters[NCLUSTERS * NFEATURES];
#pragma HLS ARRAY_PARTITION variable=local_clusters complete

    load_local_cluster(local_clusters, clusters);

    UPDATE_MEMBER1: for (ii=0; ii<NUM_TILES; ++ii)
    {
    	load_local_feature(local_feature, feature, ii);
    	compute_local_membership(local_feature, local_clusters, local_membership);
        store_local_membership(local_membership, membership, ii);
    }
}

}