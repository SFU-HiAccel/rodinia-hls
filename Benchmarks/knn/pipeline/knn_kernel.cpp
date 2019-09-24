#include <math.h>
#include <string.h>


#define BUF_SIZE_OFFSET 1            
#define BUF_SIZE ((BUF_SIZE_OFFSET) << 18)          //tile size
#define UNROLL_FACTOR 128
#define A ((BUF_SIZE_OFFSET) << 21)

void load(int i, float* local_lat, float* lat, float* local_lon, float* lon){
    #pragma HLS INLINE OFF
    memcpy(local_lat, lat + i, BUF_SIZE * sizeof(float));
    memcpy(local_lon, lon + i, BUF_SIZE * sizeof(float));
}

void compute(float* local_lon, float* local_lat, float* local_target_lat, float* local_target_lon, float* local_distance){

#pragma HLS INLINE OFF
    int m, j;
    for ( m = 0; m < BUF_SIZE; m += UNROLL_FACTOR ) {
    #pragma HLS pipeline
        for ( j = 0; j < UNROLL_FACTOR; j++ ) {                      
        #pragma HLS UNROLL FACTOR=128
            float a = local_lat[m + j] - local_target_lat[j];
            float b = local_lon[m + j] - local_target_lon[j];
            local_distance[m + j] = sqrt( a * a + b * b );
        }
    }
}

void store(int i, float* distance, float* local_distance){
#pragma HLS INLINE OFF
    memcpy(distance + i, local_distance, BUF_SIZE * sizeof(float));

}
extern "C" {
void workload(    
    float target_lat,                   // target latitude and longitude
    float target_lon,
    float* lat,                      // input latitude array
    float* lon,                      // input longitude array
    int size,                        // size of input
    float* distance
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
    #pragma HLS ARRAY_PARTITION variable=local_lat cyclic factor=128 dim=1
    float local_lon[BUF_SIZE];
    #pragma HLS ARRAY_PARTITION variable=local_lon cyclic factor=128 dim=1
    float local_dist[BUF_SIZE];
    #pragma HLS ARRAY_PARTITION variable=local_dist cyclic factor=128 dim=1

    float local_target_lat[UNROLL_FACTOR];
    #pragma HLS ARRAY_PARTITION variable=local_target_lat cyclic factor=128 dim=1

    float local_target_lon[UNROLL_FACTOR];
    #pragma HLS ARRAY_PARTITION variable=local_target_lon cyclic factor=128 dim=1

    int i;
    for( i = 0; i < UNROLL_FACTOR; i++ ) {
    #pragma HLS UNROLL FACTOR=128
        local_target_lat[i] = target_lat;
        local_target_lon[i] = target_lon;
    }


    for( i = 0; i < A; i += BUF_SIZE ){
        load(i, local_lat, lat, local_lon, lon);
        compute(local_lon, local_lat, local_target_lat, local_target_lon, local_dist);
        store(i, distance, local_dist) ;   
    }
}
}