#include <string.h>
#include <stdio.h>
#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#define NFEATURES 34
#define NPOINTS 819200
#define NCLUSTERS 5
#define TILE_SIZE 4096

#define local_clusters(i,j) local_clusters[NFEATURES*i+j]
#define local_feature(i,j) local_feature[NFEATURES*i+j]

extern "C" {
void buffer_load(int flag, float *feature, float *local_feature) {
#pragma HLS INLINE off
    if (flag) {
        memcpy(local_feature, feature, TILE_SIZE * NFEATURES * sizeof(float));
    }
}

void buffer_store(int flag, int *membership, int *local_membership) {
#pragma HLS INLINE off
    if (flag) {
        memcpy(membership, local_membership, TILE_SIZE * sizeof(int));
    }
}


void buffer_compute(int flag, float local_feature[TILE_SIZE * NFEATURES], float local_clusters[NCLUSTERS * NFEATURES],
                    int local_membership[TILE_SIZE]) {
#pragma HLS INLINE off
    if (flag) {
        int i, j, k, index;
        UPDATE_MEMBER2: for (i = 0; i < TILE_SIZE; i++) {
            //#pragma HLS PIPELINE

            float min_dist = FLT_MAX;

            /* find the cluster center id with min distance to pt */
            MIN: for (j = 0; j < NCLUSTERS; j++) {
                //#pragma HLS UNROLL
                //#pragma HLS PIPELINE

                float dist = 0.0;

                DIST: for (k = 0; k < NFEATURES; k++) {
                    //#pragma HLS UNROLL
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

void workload(float *feature, /* [npoints][nfeatures] */
              int *membership,
              float *clusters) /* [n_clusters][n_features] */
{
#pragma HLS INTERFACE m_axi port=feature offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=membership offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=clusters offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=feature bundle=control
#pragma HLS INTERFACE s_axilite port=membership bundle=control
#pragma HLS INTERFACE s_axilite port=clusters bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int i, ii, j, k, index;


    int local_membership_x[TILE_SIZE];
    int local_membership_y[TILE_SIZE];
    int local_membership_z[TILE_SIZE];

    float local_feature_x[TILE_SIZE * NFEATURES];
    #pragma HLS ARRAY_PARTITION variable=local_feature_x cyclic factor=34 dim=1

    float local_feature_y[TILE_SIZE * NFEATURES];
    #pragma HLS ARRAY_PARTITION variable=local_feature_y cyclic factor=34 dim=1

    float local_feature_z[TILE_SIZE * NFEATURES];
    #pragma HLS ARRAY_PARTITION variable=local_feature_z cyclic factor=34 dim=1

    float local_clusters[NCLUSTERS * NFEATURES];
    #pragma HLS ARRAY_PARTITION variable=local_clusters complete dim=1


    memcpy(local_clusters, clusters, NCLUSTERS * NFEATURES * sizeof(float));

    UPDATE_MEMBER1: for (i = 0; i < NPOINTS / TILE_SIZE + 2; i++) {
        int load_flag = i >= 0 && i < NPOINTS / TILE_SIZE;
        int compute_flag = i >= 1 && i < NPOINTS / TILE_SIZE + 1;
        int store_flag = i >= 2 && i < NPOINTS / TILE_SIZE + 2;

        if (i % 3 == 0) {
            buffer_load(load_flag, feature + NFEATURES * TILE_SIZE * i, local_feature_x);
            buffer_compute(compute_flag, local_feature_z, local_clusters, local_membership_z);
            buffer_store(store_flag, membership + TILE_SIZE * (i - 2), local_membership_y);
        } else if (i % 3 == 1) {
            buffer_load(load_flag, feature + NFEATURES * TILE_SIZE * i, local_feature_y);
            buffer_compute(compute_flag, local_feature_x, local_clusters, local_membership_x);
            buffer_store(store_flag, membership + TILE_SIZE * (i - 2), local_membership_z);
        } else {
            buffer_load(load_flag, feature + NFEATURES * TILE_SIZE * i, local_feature_z);
            buffer_compute(compute_flag, local_feature_y, local_clusters, local_membership_y);
            buffer_store(store_flag, membership + TILE_SIZE * (i - 2), local_membership_x);
        }
    }


}
}
