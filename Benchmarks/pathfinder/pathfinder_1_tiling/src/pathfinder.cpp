#include "pathfinder.h"
#include "../../../common/mc.h"

extern "C" {
void workload(int32_t J[ROWS * COLS], int32_t Jout[COLS]) {
  
 	#pragma HLS INTERFACE m_axi port=J offset=slave bundle=gmem
 	#pragma HLS INTERFACE m_axi port=Jout offset=slave bundle=gmem
  	#pragma HLS INTERFACE s_axilite port=J bundle=control
  	#pragma HLS INTERFACE s_axilite port=Jout bundle=control
  	#pragma HLS INTERFACE s_axilite port=return bundle=control
	
	int32_t dst[COLS], src[COLS];
	int32_t i, t, tt, n;
	int32_t min;

	int32_t wall[COLS * TILE_SIZE];
	#pragma HLS array_partition variable=wall cyclic factor=64
	#pragma HLS array_partition variable=dst cyclic factor=64
	#pragma HLS array_partition variable=src cyclic factor=64

	memcpy(dst,J,sizeof(int32_t) * COLS);

	KERNEL_OUTER: for(t = 0; t < ROWS/TILE_SIZE ;t++){
		memcpy(wall,J + COLS*(t*TILE_SIZE+1),sizeof(int32_t) * COLS * TILE_SIZE);

		KERNEL_TILE:for(tt = 0; tt < TILE_SIZE; tt++){
			KERNEL_INNER: for(n = 0; n < COLS; n++){
				min = dst[n];
			
				if(n > 0){
					min = MIN(min,dst[n-1]);
				}
	
				if(n < COLS-1){
					min = MIN(min,dst[n+1]);
				}

				if(t * TILE_SIZE + tt < ROWS-1){
					src[n] = wall[tt * COLS + n]+min;	
				}
				
			}
			memcpy(dst,src,sizeof(int32_t) * COLS);
		}
	}  	
	memcpy(Jout,dst,sizeof(int32_t) * COLS);
	return;
}
}