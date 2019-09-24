#include <math.h>
#include <string.h>

#define BUF_SIZE_OFFSET 1            
#define BUF_SIZE ((BUF_SIZE_OFFSET) << 18)          //tile size

#define BATCH_SIZE ((BUF_SIZE_OFFSET) << 21)

void load(int i, float* local_lat, float* lat, float* local_lon, float* lon){
    #pragma HLS INLINE OFF
    memcpy(local_lat, lat + i, BUF_SIZE * sizeof(float));
    memcpy(local_lon, lon + i, BUF_SIZE * sizeof(float));
}

void compute(float* local_lon, float* local_lat, float target_lat, float target_lon, float* local_distance){

#pragma HLS INLINE OFF
    int ii;
    for( ii = 0; ii < BUF_SIZE; ii++ ) {
        float a = local_lat[ii] - target_lat;
        float b = local_lon[ii] - target_lon;
        local_distance[ii] = sqrt( a * a + b * b );
    }
}

void store(int i, float* distance, float* local_distance){
#pragma HLS INLINE OFF
    memcpy(distance + i, local_distance, BUF_SIZE * sizeof(float));

}

extern "C" {
void workload(    
    float target_lat,                // target latitude
    float target_lon,                // target longitude
    float* lat,                      // input latitude array
    float* lon,                      // input longitude array
    int size,                        // size of input
    float* distance                  // output distance array
)
{
    #pragma HLS INTERFACE m_axi port=lat offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=lon offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=distance offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=target_lat bundle=control
    #pragma HLS INTERFACE s_axilite port=target_lon bundle=control
    #pragma HLS INTERFACE s_axilite port=lat bundle=control
    #pragma HLS INTERFACE s_axilite port=lon bundle=control
    #pragma HLS INTERFACE s_axilite port=size bundle=control
    #pragma HLS INTERFACE s_axilite port=distance bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    float local_lat[BUF_SIZE];
    float local_lon[BUF_SIZE];
    float local_distance[BUF_SIZE];
    
    int i;
    for( i = 0; i < BATCH_SIZE; i += BUF_SIZE ){
        load(i, local_lat, lat, local_lon, lon);
        compute(local_lon, local_lat, target_lat, target_lon, local_distance);
        store(i, distance, local_distance) ;   
    }

    return;
}

}