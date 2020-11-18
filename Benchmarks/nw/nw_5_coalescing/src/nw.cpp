#include "nw.h"
#include <string.h>
#include <inttypes.h>

#define MATCH_SCORE 1
#define MISMATCH_SCORE -1
#define GAP_SCORE -1

#define ALIGN '\\'
#define SKIPA '^'
#define SKIPB '<'

#define MAX(A,B) ( ((A)>(B))?(A):(B) )

#define JOBS_PER_BATCH 512
#define UNROLL_FACTOR 64
#define JOBS_PER_PE ((JOBS_PER_BATCH)/(UNROLL_FACTOR))

void needwun(char SEQA[ALEN], char SEQB[BLEN],
             char alignedA[ALEN+BLEN], char alignedB[ALEN+BLEN]){

    char ptr[(ALEN+1)*(BLEN+1)];

    char M_former[ALEN+1];
    #pragma HLS ARRAY_PARTITION variable=M_former dim=0 complete
    char M_latter[ALEN+1];
    #pragma HLS ARRAY_PARTITION variable=M_latter dim=0 complete

    char score, up_left, up, left, max;
    int row, row_up, r;
    int a_idx, b_idx;
    int a_str_idx, b_str_idx;

    for (a_idx=0; a_idx<ALEN+1; a_idx++) {
    #pragma HLS UNROLL
        M_former[a_idx] = a_idx*GAP_SCORE;
    }
    for (a_idx=0; a_idx<ALEN+1; a_idx++) {
    #pragma HLS PIPELINE
        ptr[a_idx] = SKIPB;
    }

    // Matrix filling loop
    fill_out: for(b_idx=1; b_idx<(BLEN+1); b_idx++){
        fill_in: for(a_idx=0; a_idx<(ALEN+1); a_idx++){
	#pragma HLS PIPELINE
      	    if (a_idx == 0) {
      	        M_latter[0] = b_idx * GAP_SCORE;
      		      ptr[b_idx*(ALEN+1)] = SKIPA;
      	    }
      	    else {
                if(SEQA[a_idx-1] == SEQB[b_idx-1]){
                    score = MATCH_SCORE;
                } else {
                    score = MISMATCH_SCORE;
                }

      	        char x = M_former[ALEN];
                char y = M_former[0  ];
                char z = M_latter[ALEN];

                up_left = x + score;
                up      = y + GAP_SCORE;
                left    = z + GAP_SCORE;

                max = MAX(up_left, MAX(up, left));

                M_latter[0] = max;

                row = (b_idx)*(ALEN+1);
                if(max == left){
                    ptr[row + a_idx] = SKIPB;
                } else if(max == up){
                    ptr[row + a_idx] = SKIPA;
                } else{
                    ptr[row + a_idx] = ALIGN;
                }
      	    }
  	    //-- shifting register
      	    char tmp_former = M_former[0];
      	    char tmp_latter = M_latter[0];

      	    for(int i=0; i<ALEN+1-1; i++){
      	        M_former[i] = M_former[i+1] ; 
      	        M_latter[i] = M_latter[i+1] ; 
      	    }

      	    M_former[ALEN+1-1] = tmp_former;
      	    M_latter[ALEN+1-1] = tmp_latter;
        }

        for (int k=0; k<ALEN+1; k++) {
        #pragma HLS UNROLL
            M_former[k] = M_latter[k];
        }
    }

    // TraceBack (n.b. aligned sequences are backwards to avoid string appending)
    a_idx = ALEN;
    b_idx = BLEN;
    a_str_idx = 0;
    b_str_idx = 0;

    trace: while(a_idx>0 || b_idx>0) {
    #pragma HLS PIPELINE
        r = b_idx*(ALEN+1);
        if (ptr[r + a_idx] == ALIGN){
            alignedA[a_str_idx++] = SEQA[a_idx-1];
            alignedB[b_str_idx++] = SEQB[b_idx-1];
            a_idx--;
            b_idx--;
        }
        else if (ptr[r + a_idx] == SKIPB){
            alignedA[a_str_idx++] = SEQA[a_idx-1];
            alignedB[b_str_idx++] = '-';
            a_idx--;
        }
        else{ // SKIPA
            alignedA[a_str_idx++] = '-';
            alignedB[b_str_idx++] = SEQB[b_idx-1];
            b_idx--;
        }
    }

    // Pad the result
    pad_a: for( ; a_str_idx<ALEN+BLEN; a_str_idx++ ) {
    #pragma HLS PIPELINE
      alignedA[a_str_idx] = '_';
    }
    pad_b: for( ; b_str_idx<ALEN+BLEN; b_str_idx++ ) {
    #pragma HLS PIPELINE
      alignedB[b_str_idx] = '_';
    }
}

void needwun_tiling(char* SEQA, char* SEQB,
             char* alignedA, char* alignedB) {
	for (int i=0; i<JOBS_PER_PE; i++) {
	    needwun(SEQA + i*ALEN, SEQB + i*BLEN,
		    alignedA + i*(ALEN+BLEN), alignedB + i*(ALEN+BLEN));
	}
	return;
}

void copy_in(uint64_t* large_A, char* part_buf_A) {
  int i, j;
  for (i=0; i<JOBS_PER_PE; i++) {
    for (j=0; j<16; j++) {
    #pragma HLS PIPELINE
      uint64_t curr = large_A[i*16+j];
      part_buf_A[i*128+j*8+0] = (uint8_t)(curr >> 0);
      part_buf_A[i*128+j*8+1] = (uint8_t)(curr >> 8);
      part_buf_A[i*128+j*8+2] = (uint8_t)(curr >> 16);
      part_buf_A[i*128+j*8+3] = (uint8_t)(curr >> 24);
      part_buf_A[i*128+j*8+4] = (uint8_t)(curr >> 32);
      part_buf_A[i*128+j*8+5] = (uint8_t)(curr >> 40);
      part_buf_A[i*128+j*8+6] = (uint8_t)(curr >> 48);
      part_buf_A[i*128+j*8+7] = (uint8_t)(curr >> 56);
    }
  }
}

void copy_out(uint64_t* large_E, char* part_buf_E) {
  int i, j;
  for (i=0; i<JOBS_PER_PE; i++) {
    for (j=0; j<16; j++) {
    #pragma HLS PIPELINE
      uint64_t curr = part_buf_E[i*128+j*8+7];
      curr = (curr << 8) | part_buf_E[i*128+j*8+6];
      curr = (curr << 8) | part_buf_E[i*128+j*8+5];
      curr = (curr << 8) | part_buf_E[i*128+j*8+4];
      curr = (curr << 8) | part_buf_E[i*128+j*8+3];
      curr = (curr << 8) | part_buf_E[i*128+j*8+2];
      curr = (curr << 8) | part_buf_E[i*128+j*8+1];
      curr = (curr << 8) | part_buf_E[i*128+j*8+0];
      large_E[i*16+j] = curr;
    }
  }
}

void buffer_load(int flag, uint64_t* global_buf_A, char part_buf_A[UNROLL_FACTOR][ALEN*JOBS_PER_PE], uint64_t* global_buf_B, char part_buf_B[UNROLL_FACTOR][BLEN*JOBS_PER_PE]) {
#pragma HLS INLINE off

  uint64_t large_A[UNROLL_FACTOR][ALEN*JOBS_PER_PE/8];
  #pragma HLS ARRAY_PARTITION variable=large_A dim=1 complete
  uint64_t large_B[UNROLL_FACTOR][BLEN*JOBS_PER_PE/8];
  #pragma HLS ARRAY_PARTITION variable=large_B dim=1 complete

  int i;
  if (flag) {
    for (i=0; i<UNROLL_FACTOR; i++) {
      memcpy(large_A[i], global_buf_A + i * (ALEN*JOBS_PER_PE)/8, ALEN*JOBS_PER_PE);
    }
    for (i=0; i<UNROLL_FACTOR; i++) {
    #pragma HLS UNROLL
      copy_in(large_A[i], part_buf_A[i]);
    }
    for (i=0; i<UNROLL_FACTOR; i++) {
      memcpy(large_B[i], global_buf_B + i * (BLEN*JOBS_PER_PE)/8, BLEN*JOBS_PER_PE);
    }
    for (i=0; i<UNROLL_FACTOR; i++) {
    #pragma HLS UNROLL
      copy_in(large_B[i], part_buf_B[i]);
    }
  }
  return;
}

void buffer_store(int flag, uint64_t* global_buf_A, char part_buf_A[UNROLL_FACTOR][(ALEN+BLEN)*JOBS_PER_PE], uint64_t* global_buf_B, char part_buf_B[UNROLL_FACTOR][(ALEN+BLEN)*JOBS_PER_PE]) {
#pragma HLS INLINE off
  uint64_t large_A[UNROLL_FACTOR][(ALEN+BLEN)*JOBS_PER_PE/8];
  #pragma HLS ARRAY_PARTITION variable=large_A dim=1 complete
  uint64_t large_B[UNROLL_FACTOR][(ALEN+BLEN)*JOBS_PER_PE/8];
  #pragma HLS ARRAY_PARTITION variable=large_B dim=1 complete
  int i;
  if (flag) {
    for (i=0; i<UNROLL_FACTOR; i++) {
    #pragma HLS UNROLL
      copy_out(large_A[i], part_buf_A[i]);
    }
    for (i=0; i<UNROLL_FACTOR; i++) {
      memcpy(global_buf_A + i * ((ALEN+BLEN)*JOBS_PER_PE)/64, large_A[i], (ALEN+BLEN)*JOBS_PER_PE);
    }
    for (i=0; i<UNROLL_FACTOR; i++) {
    #pragma HLS UNROLL
      copy_out(large_B[i], part_buf_B[i]);
    }
    for (i=0; i<UNROLL_FACTOR; i++) {
      memcpy(global_buf_B + i * ((ALEN+BLEN)*JOBS_PER_PE)/64, large_B[i], (ALEN+BLEN)*JOBS_PER_PE);
    }
  }
  return;
}

void buffer_compute(int flag, char seqA_buf[UNROLL_FACTOR][ALEN*JOBS_PER_PE],
	                      char seqB_buf[UNROLL_FACTOR][BLEN*JOBS_PER_PE],
		              char alignedA_buf[UNROLL_FACTOR][(ALEN+BLEN)*JOBS_PER_PE],      
                              char alignedB_buf[UNROLL_FACTOR][(ALEN+BLEN)*JOBS_PER_PE]) {
#pragma HLS INLINE off
  int j;
  if (flag) {
    for (j=0; j<UNROLL_FACTOR; j++) {
    #pragma HLS UNROLL
	needwun_tiling(seqA_buf[j], seqB_buf[j], alignedA_buf[j], alignedB_buf[j]);
    }
  }
  return;
}

extern "C" {
void workload(uint64_t* SEQA, uint64_t* SEQB,
              uint64_t* alignedA, uint64_t* alignedB, int num_jobs) {
#pragma HLS INTERFACE m_axi port=SEQA offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=SEQB offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=alignedA offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=alignedB offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=SEQA bundle=control
#pragma HLS INTERFACE s_axilite port=SEQB bundle=control
#pragma HLS INTERFACE s_axilite port=alignedA bundle=control
#pragma HLS INTERFACE s_axilite port=alignedB bundle=control
#pragma HLS INTERFACE s_axilite port=num_jobs bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  int num_batches = num_jobs / JOBS_PER_BATCH;

  char seqA_buf_x[UNROLL_FACTOR][ALEN * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=seqA_buf_x complete dim=1
  char seqA_buf_y[UNROLL_FACTOR][ALEN * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=seqA_buf_y complete dim=1

  char seqB_buf_x[UNROLL_FACTOR][BLEN * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=seqB_buf_x complete dim=1
  char seqB_buf_y[UNROLL_FACTOR][BLEN * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=seqB_buf_y complete dim=1

  char alignedA_buf_x[UNROLL_FACTOR][(ALEN+BLEN) * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=alignedA_buf_x complete dim=1
  char alignedA_buf_y[UNROLL_FACTOR][(ALEN+BLEN) * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=alignedA_buf_y complete dim=1

  char alignedB_buf_x[UNROLL_FACTOR][(ALEN+BLEN) * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=alignedB_buf_x complete dim=1
  char alignedB_buf_y[UNROLL_FACTOR][(ALEN+BLEN) * JOBS_PER_PE];
  #pragma HLS ARRAY_PARTITION variable=alignedB_buf_y complete dim=1

  int i;
  for (i=0; i<num_batches+2; i++) {
    int load_flag = i >= 0 && i < num_batches;
    int compute_flag = i >= 1 && i < num_batches+1;
    int store_flag = i >= 2 && i < num_batches+2;
    if (i % 2 == 0) {
      buffer_compute(compute_flag, seqA_buf_y, seqB_buf_y, alignedA_buf_y, alignedB_buf_y);
      buffer_store(store_flag, alignedA+(i-2)*(ALEN+BLEN)*JOBS_PER_BATCH/8, alignedA_buf_x, alignedB+(i-2)*(ALEN+BLEN)*JOBS_PER_BATCH/8, alignedB_buf_x);
      buffer_load(load_flag, SEQA+i*ALEN*JOBS_PER_BATCH/8, seqA_buf_x, SEQB+i*BLEN*JOBS_PER_BATCH/8, seqB_buf_x);
    } 
    else {
      buffer_compute(compute_flag, seqA_buf_x, seqB_buf_x, alignedA_buf_x, alignedB_buf_x);
      buffer_store(store_flag, alignedA+(i-2)*(ALEN+BLEN)*JOBS_PER_BATCH/8, alignedA_buf_y, alignedB+(i-2)*(ALEN+BLEN)*JOBS_PER_BATCH/8, alignedB_buf_y);
      buffer_load(load_flag, SEQA+i*ALEN*JOBS_PER_BATCH/8, seqA_buf_y, SEQB+i*BLEN*JOBS_PER_BATCH/8, seqB_buf_y);
    } 
  }
  return;
}
}
