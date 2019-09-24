#include "pathfinder.h"
#include "../../../common/mc.h"

extern "C" {
void pathfinder_kernel(int32_t *dst, int32_t *src, int32_t *wall, int32_t t){
	#pragma HLS inline off

	int32_t tt,n, min;

	KERNEL_TILE:for(tt = 0; tt < TILE_SIZE; tt++){
		KERNEL_INNER: for(n = 0; n < COLS; n++){
			#pragma HLS unroll factor=64
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

		KERNEL_SWAP:for(n = 0; n < COLS/2; n++){
			#pragma HLS unroll factor=64
			dst[2*n] = src[2*n];
			dst[2*n+1] = src[2*n+1];
		}
	}
}

void load(int32_t *wall, int32_t *J, int32_t t){
	#pragma HLS inline off

	memcpy(wall,J + COLS*(t*TILE_SIZE+1),sizeof(int32_t) * COLS * TILE_SIZE);
}

void workload(int32_t J[ROWS * COLS], int32_t Jout[COLS]) {
  
 	#pragma HLS INTERFACE m_axi port=J offset=slave bundle=gmem
 	#pragma HLS INTERFACE m_axi port=Jout offset=slave bundle=gmem
  	#pragma HLS INTERFACE s_axilite port=J bundle=control
  	#pragma HLS INTERFACE s_axilite port=Jout bundle=control
  	#pragma HLS INTERFACE s_axilite port=return bundle=control
	
	int32_t dst[COLS], src[COLS];
	int32_t t;

	int32_t wall[COLS * TILE_SIZE];
	int32_t wall2[COLS * TILE_SIZE];
	#pragma HLS array_partition variable=wall cyclic factor=64
	#pragma HLS array_partition variable=wall2 cyclic factor=64
	#pragma HLS array_partition variable=dst cyclic factor=64
	#pragma HLS array_partition variable=src cyclic factor=64

	memcpy(dst,J,sizeof(int32_t) * COLS);
	memcpy(wall,J + COLS, sizeof(int32_t) * COLS * TILE_SIZE);

	KERNEL_OUTER: for(t = 1; t < ROWS/TILE_SIZE ;t++){
		if(t % 2 == 0){
			load(wall,J,t);
			pathfinder_kernel(dst,src,wall2,t-1);
		}else if(t % 2 == 1){
			load(wall2,J,t);
			pathfinder_kernel(dst,src,wall,t-1);
		}
	}  	
	pathfinder_kernel(dst,src,wall2,ROWS/TILE_SIZE-1);

	memcpy(Jout,dst,sizeof(int32_t) * COLS);
	return;
							  
}
}