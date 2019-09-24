#include <string.h>
#include <stdio.h>
#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#define NFEATURES 34
#define NPOINTS 819200
#define NCLUSTERS 5
#define TILE_SIZE 4096

#define feature(i,j) feature[NFEATURES*i+j]
#define local_clusters(i,j) local_clusters[NFEATURES*i+j]
#define local_feature(i,j) local_feature[NFEATURES*i+j]

extern "C" {
void workload(float  *feature, /* [npoints][nfeatures] */
              int *membership,
              float  *clusters) /* [n_clusters][n_features] */
{
#pragma HLS INTERFACE m_axi port=feature offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=membership offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=clusters offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=feature bundle=control
#pragma HLS INTERFACE s_axilite port=membership bundle=control
#pragma HLS INTERFACE s_axilite port=clusters bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int i, ii, j, k, index;

    int local_membership[TILE_SIZE];

    float local_feature[TILE_SIZE * NFEATURES];
    #pragma HLS ARRAY_PARTITION variable=local_feature cyclic factor=34 dim=1

    float local_clusters[NCLUSTERS * NFEATURES];
    #pragma HLS ARRAY_PARTITION variable=local_clusters complete dim=1

    memcpy(local_clusters, clusters, NCLUSTERS * NFEATURES * sizeof(float));

    UPDATE_MEMBER1: for (ii = 0; ii < NPOINTS; ii += TILE_SIZE) {

        memcpy(local_feature, feature + NFEATURES * ii, TILE_SIZE * NFEATURES * sizeof(float));

        UPDATE_MEMBER2: for (i = 0; i < TILE_SIZE; i++) {
        #pragma HLS PIPELINE

            float min_dist = FLT_MAX;

            /* find the cluster center id with min distance to pt */
            MIN: for (j = 0; j < NCLUSTERS; j++) {
            #pragma HLS UNROLL
                float dist = 0.0;

                DIST: for (k = 0; k < NFEATURES; k++) {
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
        memcpy(membership + ii, local_membership, TILE_SIZE * sizeof(int));
    }
}
}