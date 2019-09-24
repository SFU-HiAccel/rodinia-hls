/************************************************************************************
 *  (c) Copyright 2014-2015 Falcon Computing Solutions, Inc. All rights reserved.
 *
 *  This file contains confidential and proprietary information
 *  of Falcon Computing Solutions, Inc. and is protected under U.S. and
 *  international copyright and other intellectual property laws.
 *
 ************************************************************************************/

#include "mars_wide_bus.h"
#include <math.h>
#include <string.h>
#define BUF_SIZE_OFFSET 1
#define BUF_SIZE ((BUF_SIZE_OFFSET) << 15)
#define UNROLL_FACTOR 64
#define BUF_NUM 64
#define FIRST_DIM ((BUF_SIZE) / (UNROLL_FACTOR))

void stage_0(int flag,int i, ap_uint< 512 >* lat, ap_uint< 512 >* lon, float local_lat[FIRST_DIM][UNROLL_FACTOR], float local_lon[FIRST_DIM][UNROLL_FACTOR])
{
#pragma HLS INLINE OFF
  if(flag) {
    ::memcpy_wide_bus_read_float_2d(local_lat, 0, 0, (ap_uint< 512 >*)lat, i * BUF_SIZE * sizeof(float), sizeof(float) * BUF_SIZE);
    ::memcpy_wide_bus_read_float_2d(local_lon, 0, 0, (ap_uint< 512 >*)lon, i * BUF_SIZE * sizeof(float), sizeof(float) * BUF_SIZE);
  }
}


void stage_1(int flag, float local_lat[FIRST_DIM][UNROLL_FACTOR], float local_lon[FIRST_DIM][UNROLL_FACTOR], float local_dis[FIRST_DIM][UNROLL_FACTOR], float local_target_lat[UNROLL_FACTOR], float local_target_lon[UNROLL_FACTOR])
{
  
#pragma HLS INLINE OFF
  if (flag) {
    int ii, iii;
    for (ii = 0; ii < FIRST_DIM; ii++) {
      #pragma HLS pipeline II=1
      for (iii = 0; iii < UNROLL_FACTOR; iii++) {
      #pragma HLS unroll
        float a = local_lat[ii][iii] - local_target_lat[iii];
        float b = local_lon[ii][iii] - local_target_lon[iii];
        local_dis[ii][iii] = ((float)(sqrt(a * a + b * b)));
      }
    }
  }
}


void stage_2(int flag, int i, ap_uint< 512 >* dis, float local_dis[FIRST_DIM][UNROLL_FACTOR])
{
#pragma HLS INLINE OFF

  if (flag) {
    ::memcpy_wide_bus_write_float_2d((class ap_uint< 512 >*)dis, local_dis, 0, 0, sizeof(float ) * i * BUF_SIZE, sizeof(float ) * BUF_SIZE);
  }
}


void pipeline_stage(int i, ap_uint< 512 >* dis, ap_uint< 512 >* lat, ap_uint< 512 >* lon, float local_target_lon[UNROLL_FACTOR], float local_target_lat[UNROLL_FACTOR],
                  float local_lon_stage_0[FIRST_DIM][UNROLL_FACTOR], float local_lat_stage_0[FIRST_DIM][UNROLL_FACTOR],
                  float local_lon_stage_1[FIRST_DIM][UNROLL_FACTOR], float local_lat_stage_1[FIRST_DIM][UNROLL_FACTOR], float local_dis_stage_1[FIRST_DIM][UNROLL_FACTOR],
                  float local_dis_stage_2[FIRST_DIM][UNROLL_FACTOR]
)
{
#pragma HLS INLINE OFF
  stage_0(i < BUF_NUM, i, lat, lon, local_lat_stage_0, local_lon_stage_0);
  stage_1(0 < i && i < BUF_NUM + 1, local_lat_stage_1, local_lon_stage_1, local_dis_stage_1, local_target_lat, local_target_lon);
  stage_2(1 < i, i - 2, dis, local_dis_stage_2);

}

extern "C" { 

void workload(float target_lat, float target_lon, ap_uint< 512 >* lat, ap_uint< 512 >* lon, ap_uint< 512 >* distance)
{


#pragma HLS INTERFACE m_axi port=lat offset=slave depth=131072
#pragma HLS INTERFACE m_axi port=lon offset=slave depth=131072
#pragma HLS INTERFACE m_axi port=distance offset=slave depth=131072
#pragma HLS INTERFACE s_axilite port=target_lat bundle=control
#pragma HLS INTERFACE s_axilite port=target_lon bundle=control
#pragma HLS INTERFACE s_axilite port=lat bundle=control
#pragma HLS INTERFACE s_axilite port=lon bundle=control
#pragma HLS INTERFACE s_axilite port=distance bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  
#pragma ACCEL interface variable=distance depth=2097152 max_depth=2097152
#pragma ACCEL interface variable=lon depth=2097152 max_depth=2097152
#pragma ACCEL interface variable=lat depth=2097152 max_depth=2097152
int iiii;
for(iiii = 0; iiii<1000;iiii++){
  int i;
  float local_target_lat[UNROLL_FACTOR];
  #pragma HLS array_partition variable=local_target_lat complete dim=1

  float local_target_lon[UNROLL_FACTOR];
  #pragma HLS array_partition variable=local_target_lon complete dim=1
  for (i = 0; i < UNROLL_FACTOR; i++) {
  #pragma HLS unroll
    local_target_lat[i] = target_lat;
    local_target_lon[i] = target_lon;
  }

  int count = 0;
  for (i = 0; i < BUF_NUM + 2; i++) {

    static float local_lon_2[FIRST_DIM][UNROLL_FACTOR];
    #pragma HLS array_partition variable=local_lon_2 complete dim=2
    static float local_lon_1[FIRST_DIM][UNROLL_FACTOR];
    #pragma HLS array_partition variable=local_lon_1 complete dim=2
    static float local_lon_0[FIRST_DIM][UNROLL_FACTOR];
    #pragma HLS array_partition variable=local_lon_0 complete dim=2

    static float local_lat_2[FIRST_DIM][UNROLL_FACTOR];
    #pragma HLS array_partition variable=local_lat_2 complete dim=2
    static float local_lat_1[FIRST_DIM][UNROLL_FACTOR];
    #pragma HLS array_partition variable=local_lat_1 complete dim=2
    static float local_lat_0[FIRST_DIM][UNROLL_FACTOR];
    #pragma HLS array_partition variable=local_lat_0 complete dim=2

    static float local_dis_2[FIRST_DIM][UNROLL_FACTOR];
    #pragma HLS array_partition variable=local_dis_2 complete dim=2
    static float local_dis_1[FIRST_DIM][UNROLL_FACTOR];
    #pragma HLS array_partition variable=local_dis_1 complete dim=2
    static float local_dis_0[FIRST_DIM][UNROLL_FACTOR];
    #pragma HLS array_partition variable=local_dis_0 complete dim=2

    if (count == 0) 
      pipeline_stage(i, distance ,lat, lon, local_target_lon, local_target_lat,
                  local_lon_0, local_lat_0,
                  local_lon_1, local_lat_1, local_dis_1,
                  local_dis_2
      );
    else if (count == 1) 
      pipeline_stage(i, distance ,lat, lon, local_target_lon, local_target_lat,
                  local_lon_2, local_lat_2,
                  local_lon_0, local_lat_0, local_dis_0,
                  local_dis_1 
      );
    else 
      pipeline_stage(i, distance ,lat, lon, local_target_lon, local_target_lat,
                  local_lon_1, local_lat_1,
                  local_lon_2, local_lat_2, local_dis_2,
                  local_dis_0
      );    
    count++;
    if (count == 3) 
      count = 0;
  }
}
  return;
}
}
