#include "pathfinder.h"
#include "../../../common/mc.h"

extern "C" {
void pathfinder_kernel(char J_buf[COLS*TILE_SIZE], int32_t Jout_buf1[COLS/2], int32_t Jout_buf2[COLS/2]
			,int32_t Jout_buf3[COLS/2], int32_t Jout_buf4[COLS/2]){
	#pragma HLS inline off
	/*int32_t n;
	int32_t extra = 2147483647;
	
	KERNEL_COMP: for(n = 0; n < COLS/4; n++){
		#pragma HLS pipeline II=1
		#pragma HLS unroll factor=64

		int32_t get1, get2, get3, get4;
		int32_t min1, min2, min3, min4;
		int32_t oldMin1, oldMin2, oldMin3;
		int32_t next;
		
		get1 = (int)J_buf[n*4];	
		get2 = (int)J_buf[n*4+1];
		get3 = (int)J_buf[n*4+2];
		get4 = (int)J_buf[n*4+3];

		oldMin1 = min1 = Jout_buf1[n*2];
		oldMin3 = min3 = Jout_buf1[n*2+1];
		oldMin2 = min2 = Jout_buf3[n*2];
		min4 = Jout_buf3[n*2+1];
		next = Jout_buf1[n*2+2];

		min1 = MIN(min1,extra);
		min1 = MIN(min1,min2);
		
		extra = min4;
		///////////////////////////////////////////////////////
		
		min2 = MIN(min2,oldMin1);
		min2 = MIN(min2,min3);	
		
		///////////////////////////////////////////////////////
						
		min3 = MIN(min3,oldMin2);
		min3 = MIN(min3,min4);
		
		///////////////////////////////////////////////////////
		
		min4 = MIN(min4,oldMin3);
		min4 = MIN(min4,next);	

		Jout_buf2[n*2] = get1+min1;
		Jout_buf2[n*2+1] = get3+min3;
		Jout_buf4[n*2] = get2+min2;
		Jout_buf4[n*2+1] = get4+min4; 
	}*/

	int32_t n;
	int32_t extra;
	int32_t t;

	for(t = 0; t < TILE_SIZE; t+=2){
		KERNEL_COMP_1: for(n = 0; n < COLS/2; n++){
			#pragma HLS pipeline II=1
			#pragma HLS unroll factor=64

			int32_t get1, get2;
			int32_t min1, min2;
			int32_t oldMin1, oldMin2, oldMin3;
			int32_t next;
		
			get1 = (int)J_buf[t * COLS + n*2];	
			get2 = (int)J_buf[t * COLS + n*2+1];

			oldMin1 = min1 = Jout_buf1[n];
			oldMin2 = min2 = Jout_buf3[n];
			next = Jout_buf1[n+1];
		

			if(n > 0){
				min1 = MIN(min1,extra);
			}

			min1 = MIN(min1,min2);
		
			extra = min2;
			///////////////////////////////////////////////////////
		
			min2 = MIN(min2,oldMin1);

			if(n < COLS/2){
				min2 = MIN(min2,next);
			}	
		
			///////////////////////////////////////////////////////

			Jout_buf2[n] = get1+min1;
			Jout_buf4[n] = get2+min2;
		}

		KERNEL_COMP_2: for(n = 0; n < COLS/2; n++){
			#pragma HLS pipeline II=1
			#pragma HLS unroll factor=64

			int32_t get1, get2;
			int32_t min1, min2;
			int32_t oldMin1, oldMin2, oldMin3;
			int32_t next;
		
			get1 = (int)J_buf[(t+1) * COLS + n*2];	
			get2 = (int)J_buf[(t+1) * COLS + n*2+1];

			oldMin1 = min1 = Jout_buf2[n];
			oldMin2 = min2 = Jout_buf4[n];
			next = Jout_buf2[n+1];
	
			if(n > 0){
				min1 = MIN(min1,extra);
			}

			min1 = MIN(min1,min2);
		
			extra = min2;
			///////////////////////////////////////////////////////
		
			min2 = MIN(min2,oldMin1);

			if(n < COLS/2){
				min2 = MIN(min2,next);
			}	
		
			///////////////////////////////////////////////////////

			Jout_buf1[n] = get1+min1;
			Jout_buf3[n] = get2+min2;
		}
	}
	
	return;
}

void load(char J_buf[COLS*TILE_SIZE], class ap_uint<LARGE_BUS> *J, int32_t t){
	#pragma HLS inline off

	memcpy_wide_bus_read_char(J_buf,
		((class ap_uint<LARGE_BUS> *)(J + (COLS*(t * TILE_SIZE+1))/(LARGE_BUS/8))),
		0*sizeof(char),//COLS*(t * TILE_SIZE+1)*sizeof(char),
		sizeof(char)*COLS*TILE_SIZE);
	
	
	return;
}

void workload(class ap_uint<LARGE_BUS> *J, class ap_uint<LARGE_BUS> *Jout){//, int32_t repeat) {
  
 	#pragma HLS INTERFACE m_axi port=J offset=slave bundle=gmem
 	#pragma HLS INTERFACE m_axi port=Jout offset=slave bundle=gmem
  	#pragma HLS INTERFACE s_axilite port=J bundle=control
  	#pragma HLS INTERFACE s_axilite port=Jout bundle=control
	//#pragma HLS INTERFACE s_axilite port=repeat bundle=control
  	#pragma HLS INTERFACE s_axilite port=return bundle=control
	
	char J_buf1[COLS*TILE_SIZE];
	#pragma HLS array_partition variable=J_buf1 cyclic factor=64
	char J_buf2[COLS*TILE_SIZE];
	#pragma HLS array_partition variable=J_buf2 cyclic factor=64

	int32_t Jout_buf1[COLS/2];
	#pragma HLS array_partition variable=Jout_buf1 cyclic factor=64
	int32_t Jout_buf2[COLS/2];
	#pragma HLS array_partition variable=Jout_buf2 cyclic factor=64

	int32_t Jout_buf3[COLS/2];
	#pragma HLS array_partition variable=Jout_buf3 cyclic factor=64
	int32_t Jout_buf4[COLS/2];
	#pragma HLS array_partition variable=Jout_buf4 cyclic factor=64

	int32_t Jtemp[COLS*TILE_SIZE/4];
	#pragma HLS array_partition variable=Jtemp cyclic factor=64

	int32_t repeatCount = 0;
	//for(repeatCount = 0; repeatCount < repeat; repeatCount++){

	memcpy_wide_bus_read_int(Jtemp,
		((class ap_uint<LARGE_BUS> *)J),
		0,
		sizeof(char) * COLS*TILE_SIZE);

	int32_t t;

	KERNEL_TRANSFER: for(t = 0; t < COLS/4; t++){
		#pragma HLS pipeline II=1
		#pragma HLS unroll factor=16
		Jout_buf3[t * 2 + 1] = (Jtemp[t] >> 24) & 0xFF;
		Jout_buf1[t * 2 + 1] = (Jtemp[t] >> 16) & 0xFF;
		Jout_buf3[t * 2] = (Jtemp[t] >> 8) & 0xFF;
		Jout_buf1[t * 2] = Jtemp[t] & 0xFF;
	}

	memcpy_wide_bus_read_char(J_buf2,
		((class ap_uint<LARGE_BUS> *)J),
		COLS*sizeof(char),
		sizeof(char)*COLS*TILE_SIZE);
	
	KERNEL_OUTER: for(t = 1; t < ROWS/TILE_SIZE;t++){
		if(t % 2 == 0){
			load(J_buf2,J,t);
			pathfinder_kernel(J_buf1,Jout_buf1,Jout_buf2,Jout_buf3,Jout_buf4);
		}else if(t % 2 == 1){
			load(J_buf1,J,t);
			pathfinder_kernel(J_buf2,Jout_buf1,Jout_buf2,Jout_buf3,Jout_buf4);
		}	
	}

	pathfinder_kernel(J_buf1,Jout_buf1,Jout_buf2,Jout_buf3,Jout_buf4);

	KERNEL_COPY: for(t = 0; t < COLS/2; t++){
		#pragma HLS unroll factor=32
		Jtemp[t*2] = Jout_buf2[t];
		Jtemp[t*2+1] = Jout_buf4[t];
	}

	memcpy_wide_bus_write_int(((class ap_uint<LARGE_BUS> *)Jout),
		Jtemp,
		0,
		sizeof(int32_t)*COLS);

	//}
  	return;
							  
}
}