#include <math.h>
#define OFFSET 1
#define BATCH_SIZE (OFFSET<<21)
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
    

    int i;
    for( i = 0; i < BATCH_SIZE; i++ ) {
        float a = lat[i] - target_lat;
        float b = lon[i] - target_lon;
        distance[i] = sqrt( a * a + b * b );
    }

    return;
}
}