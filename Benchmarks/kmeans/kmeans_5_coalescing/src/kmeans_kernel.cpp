#define __kernel
#define __global

#include <string.h>
#ifndef LARGE_BUS
#define LARGE_BUS 512
#endif
#define MARS_WIDE_BUS_TYPE ap_uint<LARGE_BUS>
#include "ap_int.h"
#include "../../../common/mars_wide_bus.h"
#define SIZE_1 48
#include "../../../common/mars_wide_bus_2d.h"
#undef SIZE_1

void mars_kernel_0_1_node_1_stage_0(int ii,int exec,class ap_uint< 512 > *feature,float feature_buf_0[4096][48])
{
#pragma HLS INLINE OFF
    if (exec == 1) {
        memcpy_wide_bus_read_float_2d_48(feature_buf_0,((size_t )0),((size_t )0),((class ap_uint< 512 > *)feature),((unsigned long )(ii * 196608)) * sizeof(float ),sizeof(float ) * ((unsigned long )196594));
    }
}

void mars_kernel_0_1_node_4_stage_2(int ii,int exec,class ap_uint< 512 > *membership,int membership_buf_0[4096])
{
#pragma HLS INLINE OFF
    if (exec == 1) {
        memcpy_wide_bus_write_int(((class ap_uint< 512 > *)membership),membership_buf_0,sizeof(int ) * ((unsigned long )(ii * 4096)),sizeof(int ) * ((unsigned long )4096));
    }
}

void mars_kernel_0_1_bus(int mars_ii,int mars_init,int mars_cond,int *mars__in_ii_0,int *mars__in_ii_1,float mars_clusters_buf_0_1[5][34],class ap_uint< 512 > *feature,float mars_feature_buf_0_0[4096][48],float mars_feature_buf_0_1[4096][48],int *mars_index_1,class ap_uint< 512 > *membership,int mars_membership_buf_0_1[4096],int mars_membership_buf_0_2[4096])
{
#pragma HLS INLINE OFF
    mars_kernel_0_1_node_1_stage_0(mars_ii - 0,((int )((mars_ii >= mars_init + 0) & (mars_ii <= mars_cond + 0))),feature,mars_feature_buf_0_0);
    mars_kernel_0_1_node_4_stage_2(mars_ii - 2,((int )((mars_ii >= mars_init + 2) & (mars_ii <= mars_cond + 2))),membership,mars_membership_buf_0_2);
}

void mars_kernel_0_1_node_3_stage_1(int ii,int exec,int *_in_ii,float clusters_buf_0[5][34],float feature_buf_0[4096][48],int *index,int membership_buf_0[4096])
{
#pragma HLS INLINE OFF
    if (exec == 1) {
        int k;
        int j;
        int i;
        for (i = 0; i < 4096; i++)
        {
        #pragma HLS dependence variable=membership_buf_0 array inter false

        #pragma HLS dependence variable=feature_buf_0 array inter false

        #pragma HLS dependence variable=clusters_buf_0 array inter false

        #pragma HLS pipeline II=1

            float min_dist = (float )3.40282347e+38;
            for (j = 0; j < 5; j++)
            {
            #pragma HLS unroll
                float dist = (float )0.0;
                for (k = 0; k < 34; k++)
                {
                #pragma HLS unroll
                    float diff = feature_buf_0[i][k] - clusters_buf_0[j][k];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    *index = j;
                }
            }
            /* assign the membership to object i */
            membership_buf_0[i] =  *index;
        }
    }
}

void mars_kernel_0_1_comp(int mars_ii,int mars_init,int mars_cond,int *mars__in_ii_0,int *mars__in_ii_1,float mars_clusters_buf_0_1[5][34],class ap_uint< 512 > *feature,float mars_feature_buf_0_0[4096][48],float mars_feature_buf_0_1[4096][48],int *mars_index_1,class ap_uint< 512 > *membership,int mars_membership_buf_0_1[4096],int mars_membership_buf_0_2[4096])
{

#pragma HLS INLINE OFF
    mars_kernel_0_1_node_3_stage_1(mars_ii - 1,((int )((mars_ii >= mars_init + 1) & (mars_ii <= mars_cond + 1))),mars__in_ii_1,mars_clusters_buf_0_1,mars_feature_buf_0_1,mars_index_1,mars_membership_buf_0_1);
}

void mars_kernel_0_1(int mars_ii,int mars_init,int mars_cond,int *mars__in_ii_0,int *mars__in_ii_1,float mars_clusters_buf_0_1[5][34],class ap_uint< 512 > *feature,float mars_feature_buf_0_0[4096][48],float mars_feature_buf_0_1[4096][48],int *mars_index_1,class ap_uint< 512 > *membership,int mars_membership_buf_0_1[4096],int mars_membership_buf_0_2[4096])
{

#pragma HLS INLINE OFF
    mars_kernel_0_1_bus(mars_ii,mars_init,mars_cond,mars__in_ii_0,mars__in_ii_1,mars_clusters_buf_0_1,feature,mars_feature_buf_0_0,mars_feature_buf_0_1,mars_index_1,membership,mars_membership_buf_0_1,mars_membership_buf_0_2);
    mars_kernel_0_1_comp(mars_ii,mars_init,mars_cond,mars__in_ii_0,mars__in_ii_1,mars_clusters_buf_0_1,feature,mars_feature_buf_0_0,mars_feature_buf_0_1,mars_index_1,membership,mars_membership_buf_0_1,mars_membership_buf_0_2);
}

static void merlin_memcpy_0(float dst[5][34],int dst_idx_0,int dst_idx_1,float src[170],int src_idx_0,unsigned int len)
{
#pragma HLS inline off
    long long i;
    long long total_offset1 = (0 * 5 + dst_idx_0) * 34 + dst_idx_1;
    long long total_offset2 = 0 * 170 + src_idx_0;
    for (i = 0; i < len / 4; ++i) {

    #pragma HLS PIPELINE II=1
        dst[(total_offset1 + i) / 34][(total_offset1 + i) % 34] = src[total_offset2 + i];
    }
}
extern "C" {

__kernel void workload(class ap_uint< 512 > *feature,class ap_uint< 512 > *membership,float clusters[5 * 34]) {
#pragma HLS INTERFACE m_axi port=feature offset=slave bundle=feature depth=2457600
#pragma HLS INTERFACE m_axi port=membership offset=slave bundle=membership depth=51200
#pragma HLS INTERFACE m_axi port=clusters offset=slave bundle=clusters depth=170


#pragma HLS INTERFACE s_axilite port=feature bundle=control
#pragma HLS INTERFACE s_axilite port=membership bundle=control
#pragma HLS INTERFACE s_axilite port=clusters bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

        float clusters_buf_0[5][34];
#pragma HLS array_partition variable=clusters_buf_0 complete dim=2
#pragma HLS array_partition variable=clusters_buf_0 complete dim=1

        merlin_memcpy_0(clusters_buf_0, 0, 0, clusters, 0, sizeof(float) * ((unsigned long) 170));

        int i;
        int ii;
        int j;
        int k;
        int index;
        int mars_count_0_1 = 0;
        UPDATE_MEMBER1:
        for (ii = 0; ii <= 199 + 2; ii++) {
            int mars_kernel_0_1_membership_buf_0_2[4096];
#pragma HLS array_partition variable=mars_kernel_0_1_membership_buf_0_2 cyclic factor=16 dim=1

            int mars_kernel_0_1_membership_buf_0_1[4096];
#pragma HLS array_partition variable=mars_kernel_0_1_membership_buf_0_1 cyclic factor=16 dim=1

            int mars_kernel_0_1_membership_buf_0_0[4096];
#pragma HLS array_partition variable=mars_kernel_0_1_membership_buf_0_0 cyclic factor=16 dim=1

            float mars_kernel_0_1_feature_buf_0_2[4096][48];
#pragma HLS array_partition variable=mars_kernel_0_1_feature_buf_0_2 complete dim=2

            float mars_kernel_0_1_feature_buf_0_1[4096][48];
#pragma HLS array_partition variable=mars_kernel_0_1_feature_buf_0_1 complete dim=2

            float mars_kernel_0_1_feature_buf_0_0[4096][48];
#pragma HLS array_partition variable=mars_kernel_0_1_feature_buf_0_0 complete dim=2

            int mars_kernel_0_1__in_ii_2;
            int mars_kernel_0_1__in_ii_1;
            int mars_kernel_0_1__in_ii_0;
            if (mars_count_0_1 == 0)
                mars_kernel_0_1(ii, 0, 199, &mars_kernel_0_1__in_ii_0, &mars_kernel_0_1__in_ii_1, clusters_buf_0,
                                feature, mars_kernel_0_1_feature_buf_0_0, mars_kernel_0_1_feature_buf_0_1, &index,
                                membership, mars_kernel_0_1_membership_buf_0_0, mars_kernel_0_1_membership_buf_0_1);
            else if (mars_count_0_1 == 1)
                mars_kernel_0_1(ii, 0, 199, &mars_kernel_0_1__in_ii_2, &mars_kernel_0_1__in_ii_0, clusters_buf_0,
                                feature, mars_kernel_0_1_feature_buf_0_2, mars_kernel_0_1_feature_buf_0_0, &index,
                                membership, mars_kernel_0_1_membership_buf_0_2, mars_kernel_0_1_membership_buf_0_0);
            else
                mars_kernel_0_1(ii, 0, 199, &mars_kernel_0_1__in_ii_1, &mars_kernel_0_1__in_ii_2, clusters_buf_0,
                                feature, mars_kernel_0_1_feature_buf_0_1, mars_kernel_0_1_feature_buf_0_2, &index,
                                membership, mars_kernel_0_1_membership_buf_0_1, mars_kernel_0_1_membership_buf_0_2);
            mars_count_0_1++;
            if (mars_count_0_1 == 3)
                mars_count_0_1 = 0;
        }
        ii = 815104 + ((int) 4096LL);
}
}
