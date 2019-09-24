#include "pathfinder.h"
#include "../../../common/mc.h"

void pathfinder_kernel(int32_t J[ROWS*COLS], int32_t Jout[COLS]){
	int32_t temp;
	int32_t i, t, tt, n;
	int32_t min, before;

	KERNEL2: for(t = 0; t < ROWS-1 ;t++){
		for(n = 0; n < COLS; n++){
			min = Jout[n];
			
			if(n > 0){
				min = MIN(min,before);
			}
	
			if(n < COLS-1){
				min = MIN(min,Jout[n+1]);
			}
			before = Jout[n];
			
			Jout[n] = J[(t+1) * COLS + n]+min;
		}
	}
}

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

	memcpy(dst,J,sizeof(int32_t) * COLS);

	KERNEL_OUTER: for(t = 0; t < ROWS-1 ;t++){
		KERNEL_INNER: for(n = 0; n < COLS; n++){
			min = dst[n];
			
			if(n > 0){
				min = MIN(min,dst[n-1]);
			}
	
			if(n < COLS-1){
				min = MIN(min,dst[n+1]);
			}
			
			src[n] = J[(t+1) * COLS + n]+min;
		}
		memcpy(dst,src,sizeof(int32_t) * COLS);
	}  	
	memcpy(Jout,dst,sizeof(int32_t) * COLS);
	return;
							  
}
}