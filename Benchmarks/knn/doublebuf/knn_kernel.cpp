/************************************************************************************
 *  (c) Copyright 2014-2015 Falcon Computing Solutions, Inc. All rights reserved.
 *
 *  This file contains confidential and proprietary information
 *  of Falcon Computing Solutions, Inc. and is protected under U.S. and
 *  international copyright and other intellectual property laws.
 *
 ************************************************************************************/

#include <string.h>


#include <math.h>

// Original: #pragma ACCEL kernel name="run"
void mars_kernel_0_2_node_1_stage_0(int i,int exec,float *lon,float lon_buf_0[512][128])
{
  
#pragma HLS INLINE OFF
  if (exec == 1) {
    memcpy(lon_buf_0, lon + i * 65536, 65536 * sizeof(float));
  }
// Existing HLS partition: #pragma HLS array_partition variable=lon_buf_0 cyclic factor = 16 dim=2
}

void mars_kernel_0_2_node_2_stage_0(int i,int exec,float *lat,float lat_buf_0[512][128])
{
  
#pragma HLS INLINE OFF
  if (exec == 1) {
    memcpy(lat_buf_0, lat + i * 65536, 65536 * sizeof(float));
  }
// Existing HLS partition: #pragma HLS array_partition variable=lat_buf_0 cyclic factor = 16 dim=2
}

void mars_kernel_0_2_node_6_stage_2(int i,int exec,float *distance,float distance_buf_0[512][128])
{
  
#pragma HLS INLINE OFF
  if (exec == 1) {
    memcpy(distance + i * 65536, distance_buf_0, 65536 * sizeof(float));
  }
// Existing HLS partition: #pragma HLS array_partition variable=distance_buf_0 cyclic factor = 16 dim=2
}

void mars_kernel_0_2_node_4_stage_1(int exec,float distance_buf_0[512][128],float lat_buf_0[512][128],float local_target_lat[128],float local_target_lon[128],float lon_buf_0[512][128])
{
  
#pragma HLS INLINE OFF
  if (exec == 1) {
    int iii;
    int ii;
    for (ii = 0; ii <= 511; ii++) 
// Original: #pragma ACCEL pipeline
// Original: #pragma ACCEL PIPELINE II=1
// Original: #pragma ACCEL PIPELINE II=1
{
      
#pragma HLS dependence variable=lon_buf_0 array inter false
      
#pragma HLS dependence variable=local_target_lon array inter false
      
#pragma HLS dependence variable=local_target_lat array inter false
      
#pragma HLS dependence variable=lat_buf_0 array inter false
      
#pragma HLS dependence variable=distance_buf_0 array inter false
      
#pragma HLS pipeline II=1
      for (iii = 0; iii < 128; iii++) 
// Original: #pragma ACCEL parallel flatten
// Original: #pragma ACCEL PARALLEL COMPLETE
// Original: #pragma ACCEL PARALLEL COMPLETE
{
        
#pragma HLS unroll
        float a = lat_buf_0[ii][iii] - local_target_lat[iii];
        float b = lon_buf_0[ii][iii] - local_target_lon[iii];
        distance_buf_0[ii][iii] = ((float )(sqrt(((double )(a * a + b * b)))));
      }
    }
// Stantardize from: for(ii = 0;ii < 1 << 16;ii += 128) {...}
  }
}



void mars_kernel_0_2(int mars_i,int mars_init,int mars_cond,float *distance,float mars_distance_buf_0_1[512][128],float mars_distance_buf_0_2[512][128],float *lat,float mars_lat_buf_0_0[512][128],float mars_lat_buf_0_1[512][128],float mars_local_target_lat_1[128],float mars_local_target_lon_1[128],float *lon,float mars_lon_buf_0_0[512][128],float mars_lon_buf_0_1[512][128])
{
  
#pragma HLS INLINE OFF
  mars_kernel_0_2_node_1_stage_0(mars_i,((int )((mars_i >= mars_init) & (mars_i <= mars_cond))),lon,mars_lon_buf_0_0);
  mars_kernel_0_2_node_2_stage_0(mars_i,((int )((mars_i >= mars_init) & (mars_i <= mars_cond))),lat,mars_lat_buf_0_0);
  mars_kernel_0_2_node_6_stage_2(mars_i - 2,((int )((mars_i >= mars_init + 2) & (mars_i <= mars_cond + 2))),distance,mars_distance_buf_0_2);
  mars_kernel_0_2_node_4_stage_1(((int )((mars_i >= mars_init + 1) & (mars_i <= mars_cond + 1))),mars_distance_buf_0_1,mars_lat_buf_0_1,mars_local_target_lat_1,mars_local_target_lon_1,mars_lon_buf_0_1);
}

extern "C" {
void workload(float target_lat,float target_lon,float *lat,float *lon,int size,float *distance)
{



#pragma HLS INTERFACE m_axi port=lat offset=slave bundle=lat depth=131072
#pragma HLS INTERFACE m_axi port=lon offset=slave bundle=lon depth=131072
#pragma HLS INTERFACE m_axi port=distance offset=slave bundle=distance depth=131072


#pragma HLS INTERFACE s_axilite port=target_lat bundle=control
#pragma HLS INTERFACE s_axilite port=target_lon bundle=control
#pragma HLS INTERFACE s_axilite port=lat bundle=control
#pragma HLS INTERFACE s_axilite port=lon bundle=control
#pragma HLS INTERFACE s_axilite port=size bundle=control
#pragma HLS INTERFACE s_axilite port=distance bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  
#pragma ACCEL interface variable=distance max_depth=2097152 depth=2097152
  
#pragma ACCEL interface variable=lon max_depth=2097152 depth=2097152
  
#pragma ACCEL interface variable=lat max_depth=2097152 depth=2097152
  int i;
  float local_target_lat[128];
  
#pragma HLS array_partition variable=local_target_lat complete dim=1
  float local_target_lon[128];
  
#pragma HLS array_partition variable=local_target_lon complete dim=1
  Loop_1:
  for (i = 0; i < 128; i++) 
// Original: #pragma ACCEL parallel flatten
// Original: #pragma ACCEL PARALLEL COMPLETE
// Original: #pragma ACCEL PARALLEL COMPLETE
  {
    
    #pragma HLS unroll
    local_target_lat[i] = target_lat;
    local_target_lon[i] = target_lon;
  }

  // Stantardize from: for(ii = 0;ii < 1 << 16;ii += 128) {...}
    float mars_kernel_0_2_lon_buf_0_2[512][128];
// Existing HLS partition: #pragma HLS array_partition variable=mars_kernel_0_2_lon_buf_0_2 cyclic factor=16 dim=2
    
#pragma HLS array_partition variable=mars_kernel_0_2_lon_buf_0_2 complete dim=2
    float mars_kernel_0_2_lon_buf_0_1[512][128];
// Existing HLS partition: #pragma HLS array_partition variable=mars_kernel_0_2_lon_buf_0_1 cyclic factor=16 dim=2
    
#pragma HLS array_partition variable=mars_kernel_0_2_lon_buf_0_1 complete dim=2
    float mars_kernel_0_2_lon_buf_0_0[512][128];
// Existing HLS partition: #pragma HLS array_partition variable=mars_kernel_0_2_lon_buf_0_0 cyclic factor=16 dim=2
    
#pragma HLS array_partition variable=mars_kernel_0_2_lon_buf_0_0 complete dim=2
    float mars_kernel_0_2_lat_buf_0_2[512][128];
    
#pragma HLS array_partition variable=mars_kernel_0_2_lat_buf_0_2 complete dim=2
    float mars_kernel_0_2_lat_buf_0_1[512][128];
    
#pragma HLS array_partition variable=mars_kernel_0_2_lat_buf_0_1 complete dim=2
    float mars_kernel_0_2_lat_buf_0_0[512][128];
    
#pragma HLS array_partition variable=mars_kernel_0_2_lat_buf_0_0 complete dim=2
    float mars_kernel_0_2_distance_buf_0_2[512][128];
// Existing HLS partition: #pragma HLS array_partition variable=mars_kernel_0_2_distance_buf_0_2 cyclic factor=16 dim=2
    
#pragma HLS array_partition variable=mars_kernel_0_2_distance_buf_0_2 complete dim=2
    float mars_kernel_0_2_distance_buf_0_1[512][128];
// Existing HLS partition: #pragma HLS array_partition variable=mars_kernel_0_2_distance_buf_0_1 cyclic factor=16 dim=2
    
#pragma HLS array_partition variable=mars_kernel_0_2_distance_buf_0_1 complete dim=2
    float mars_kernel_0_2_distance_buf_0_0[512][128];
// Existing HLS partition: #pragma HLS array_partition variable=mars_kernel_0_2_distance_buf_0_0 cyclic factor=16 dim=2
    
#pragma HLS array_partition variable=mars_kernel_0_2_distance_buf_0_0 complete dim=2
// Stantardize from: for(i = 0;i < 2097152;i += 1 << 16) {...}
  int mars_count_0_2 = 0;
  for (i = 0; i <= 33; i++) 
// Original: #pragma ACCEL pipeline
// Original: #pragma ACCEL PIPELINE II=1
// Original: #pragma ACCEL PIPELINE II=1
  {
    int mars_kernel_0_2__in_i_0;
    if (mars_count_0_2 == 0)
      mars_kernel_0_2(i,0,31,distance,mars_kernel_0_2_distance_buf_0_0,mars_kernel_0_2_distance_buf_0_1,lat,mars_kernel_0_2_lat_buf_0_0,mars_kernel_0_2_lat_buf_0_1,local_target_lat,local_target_lon,lon,mars_kernel_0_2_lon_buf_0_0,mars_kernel_0_2_lon_buf_0_1);
     else if (mars_count_0_2 == 1) 
      mars_kernel_0_2(i,0,31,distance,mars_kernel_0_2_distance_buf_0_2,mars_kernel_0_2_distance_buf_0_0,lat,mars_kernel_0_2_lat_buf_0_2,mars_kernel_0_2_lat_buf_0_0,local_target_lat,local_target_lon,lon,mars_kernel_0_2_lon_buf_0_2,mars_kernel_0_2_lon_buf_0_0);
     else 
      mars_kernel_0_2(i,0,31,distance,mars_kernel_0_2_distance_buf_0_1,mars_kernel_0_2_distance_buf_0_2,lat,mars_kernel_0_2_lat_buf_0_1,mars_kernel_0_2_lat_buf_0_2,local_target_lat,local_target_lon,lon,mars_kernel_0_2_lon_buf_0_1,mars_kernel_0_2_lon_buf_0_2);
    mars_count_0_2++;
    if (mars_count_0_2 == 3) 
      mars_count_0_2 = 0;
  }
  return ;
// Original label: Loop_2:
// Original label: Loop_3:

}

}


