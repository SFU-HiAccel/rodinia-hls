#include "nw.h"

#define MATCH_SCORE 1
#define MISMATCH_SCORE -1
#define GAP_SCORE -1

#define ALIGN '\\'
#define SKIPA '^'
#define SKIPB '<'

#define MAX(A,B) ( ((A)>(B))?(A):(B) )

#define JOBS_PER_BATCH 256
//#define UNROLL_FACTOR 1
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
 //trace: for (a_idx = ALEN, b_idx = BLEN; a_idx>0 || b_idx>0; ) {
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

extern "C" {
void workload(char* SEQA, char* SEQB,
             char* alignedA, char* alignedB, int num_jobs) {
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

	char seqA_buf[UNROLL_FACTOR][ALEN * JOBS_PER_PE];
	#pragma HLS ARRAY_PARTITION variable=seqA_buf complete dim=1
	char seqB_buf[UNROLL_FACTOR][BLEN * JOBS_PER_PE];
	#pragma HLS ARRAY_PARTITION variable=seqB_buf complete dim=1
	char alignedA_buf[UNROLL_FACTOR][(ALEN+BLEN) * JOBS_PER_PE];
	#pragma HLS ARRAY_PARTITION variable=alignedA_buf complete dim=1
	char alignedB_buf[UNROLL_FACTOR][(ALEN+BLEN) * JOBS_PER_PE];
	#pragma HLS ARRAY_PARTITION variable=alignedB_buf complete dim=1

	int num_batches = num_jobs / JOBS_PER_BATCH;

	int i, j, k;
	major_loop: for (i=0; i<num_batches; i++) {
	    // step 1: copy data in
	    reshape1_a: for (j=0; j<UNROLL_FACTOR; j++) {
	        memcpy(seqA_buf[j], SEQA + i*(ALEN*JOBS_PER_BATCH) + j*(ALEN*JOBS_PER_PE), ALEN*JOBS_PER_PE);
	    }
	    reshape1_b: for (j=0; j<UNROLL_FACTOR; j++) {
	        memcpy(seqB_buf[j], SEQB + i*(BLEN*JOBS_PER_BATCH) + j*(BLEN*JOBS_PER_PE), BLEN*JOBS_PER_PE);
	    }
	    // step 2: do the jobs
	    unroll_loop: for (j=0; j<UNROLL_FACTOR; j++) {
            #pragma HLS UNROLL
            needwun_tiling(seqA_buf[j], seqB_buf[j], alignedA_buf[j], alignedB_buf[j]);
	    }
	    // step 3: copy results back
	    reshape2_a: for (j=0; j<UNROLL_FACTOR; j++) {
	        memcpy(alignedA + i*((ALEN+BLEN)*JOBS_PER_BATCH) + j*(ALEN+BLEN)*JOBS_PER_PE, alignedA_buf[j], (ALEN+BLEN)*JOBS_PER_PE);
	    }
	    reshape2_b: for (j=0; j<UNROLL_FACTOR; j++) {
	        memcpy(alignedB + i*((ALEN+BLEN)*JOBS_PER_BATCH) + j*(ALEN+BLEN)*JOBS_PER_PE, alignedB_buf[j], (ALEN+BLEN)*JOBS_PER_PE);
	    }
	}
	return;
}

}
