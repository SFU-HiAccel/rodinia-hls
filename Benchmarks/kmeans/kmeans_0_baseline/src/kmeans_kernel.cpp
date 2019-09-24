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

    int i, j, k, index;

    float local_clusters[NCLUSTERS * NFEATURES];


    UPDATE_MEMBER: for (i = 0; i < NPOINTS; i++) {

        float min_dist = FLT_MAX;

        /* find the cluster center id with min distance to pt */
        MIN: for (j = 0; j < NCLUSTERS; j++) {
            float dist = 0.0;

            DIST: for (k = 0; k < NFEATURES; k++) {
                float diff = feature[NFEATURES * i + k] - clusters[NFEATURES * j + k];
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                index = j;
            }
        }

        /* assign the membership to object i */
        membership[i] = index;


    }

}
}