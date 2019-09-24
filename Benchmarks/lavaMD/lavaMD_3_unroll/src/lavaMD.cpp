#include "lavaMD.h"

void lavaMD_unrolling_padded(TYPE pos_i[N_PADDED*POS_DIM], TYPE q_i[N_PADDED], TYPE pos_o[N*POS_DIM])
{
    //local variables
    int i, j, k;
    int ii, jj, kk;
    int x, y, z;
    int l, m, n;
    int x_n, y_n, z_n;
    int A_idx, B_idx, C_idx;
    int remainder;

    TYPE r2[UNROLL_SIZE];
#pragma HLS ARRAY_PARTITION variable=r2 complete dim=0
    TYPE u2[UNROLL_SIZE];
#pragma HLS ARRAY_PARTITION variable=u2 complete dim=0
    TYPE fs[UNROLL_SIZE];
#pragma HLS ARRAY_PARTITION variable=fs complete dim=0
    TYPE vij[UNROLL_SIZE];
#pragma HLS ARRAY_PARTITION variable=vij complete dim=0
    TYPE fxij[UNROLL_SIZE];
#pragma HLS ARRAY_PARTITION variable=fxij complete dim=0
    TYPE fyij[UNROLL_SIZE];
#pragma HLS ARRAY_PARTITION variable=fyij complete dim=0
    TYPE fzij[UNROLL_SIZE];
#pragma HLS ARRAY_PARTITION variable=fzij complete dim=0
    TYPE d[UNROLL_SIZE][POS_DIM];
#pragma HLS ARRAY_PARTITION variable=d complete dim=0

    int neighborOffset[FULL_NEIGHBOR_COUNT][3] = {{-1,-1,-1}, { 0,-1,-1}, { 1,-1,-1},
                                                  {-1, 0,-1}, { 0, 0,-1}, { 1, 0,-1},
                                                  {-1, 1,-1}, { 0, 1,-1}, { 1, 1,-1},
                                                  {-1,-1, 0}, { 0,-1, 0}, { 1,-1, 0},
                                                  {-1, 0, 0}, { 0, 0, 0}, { 1, 0, 0},
                                                  {-1, 1, 0}, { 0, 1, 0}, { 1, 1, 0},
                                                  {-1,-1, 1}, { 0,-1, 1}, { 1,-1, 1},
                                                  {-1, 0, 1}, { 0, 0, 1}, { 1, 0, 1},
                                                  {-1, 1, 1}, { 0, 1, 1}, { 1, 1, 1}};
#pragma HLS ARRAY_PARTITION variable=neighborOffset complete dim=0
    //local BRAM - tiling essentials
    TYPE local_A_pos_i[NUMBER_PAR_PER_BOX][POS_DIM];
#pragma HLS ARRAY_PARTITION variable=local_A_pos_i complete dim=0
    TYPE local_B_pos_i[NUMBER_PAR_PER_BOX][POS_DIM];
#pragma HLS ARRAY_PARTITION variable=local_B_pos_i complete dim=0
    TYPE local_B_q_i[NUMBER_PAR_PER_BOX];
#pragma HLS ARRAY_PARTITION variable=local_B_q_i complete dim=0
    TYPE local_pos_o[NUMBER_PAR_PER_BOX][POS_DIM];
#pragma HLS ARRAY_PARTITION variable=local_pos_o complete dim=0	

    //iterate through all boxes in the 3D space
    BOXES: for(i=0; i<DIMENSION_3D; i++){
        //purge local_pos_o
        WIPE_LOCAL_POS_O: for (ii=0; ii<NUMBER_PAR_PER_BOX; ++ii){
#pragma HLS PIPELINE
            local_pos_o[ii][V] = 0.0;
            local_pos_o[ii][X] = 0.0;
            local_pos_o[ii][Y] = 0.0;
            local_pos_o[ii][Z] = 0.0;
        }
        //index of the output array
        C_idx = i * NUMBER_PAR_PER_BOX;
        //convert from 1D-index to 3D-index
        z = i / DIMENSION_2D;
        remainder = i % DIMENSION_2D;
        y = remainder / DIMENSION_1D;
        x = remainder % DIMENSION_1D;
        //pos of the input_A array
        x_n = x + 1;
        y_n = y + 1;
        z_n = z + 1;
        //convert from 3D-index to 1D-index
        A_idx = z_n*DIMENSION_2D_PADDED + y_n*DIMENSION_1D_PADDED + x_n;
        A_idx = A_idx * NUMBER_PAR_PER_BOX;
        //load local_A_pos_i
        LOAD_LOCAL_A_POS_I: for (ii=0; ii<NUMBER_PAR_PER_BOX; ++ii){
#pragma HLS PIPELINE II=2
            local_A_pos_i[ii][V] = pos_i[(A_idx+ii)*POS_DIM+V];
            local_A_pos_i[ii][X] = pos_i[(A_idx+ii)*POS_DIM+X];
            local_A_pos_i[ii][Y] = pos_i[(A_idx+ii)*POS_DIM+Y];
            local_A_pos_i[ii][Z] = pos_i[(A_idx+ii)*POS_DIM+Z];
        }
        //go through 27 neighbors
        for (l=0; l<FULL_NEIGHBOR_COUNT; ++l){
            //pos of the neighbor - input_B array
            x = x_n + neighborOffset[l][0];
            y = y_n + neighborOffset[l][1];
            z = z_n + neighborOffset[l][2];
            //convert from 3D-index to 1D-index
            B_idx = z*DIMENSION_2D_PADDED + y*DIMENSION_1D_PADDED + x;
            B_idx = B_idx * NUMBER_PAR_PER_BOX;
            //load local_B_pos_i
            LOAD_LOCAL_B_POS_I: for (ii=0; ii<NUMBER_PAR_PER_BOX; ++ii){
#pragma HLS PIPELINE II=2
                local_B_pos_i[ii][V] = pos_i[(B_idx+ii)*POS_DIM+V];
                local_B_pos_i[ii][X] = pos_i[(B_idx+ii)*POS_DIM+X];
                local_B_pos_i[ii][Y] = pos_i[(B_idx+ii)*POS_DIM+Y];
                local_B_pos_i[ii][Z] = pos_i[(B_idx+ii)*POS_DIM+Z];
            }
            //load local_B_q_i
            LOAD_LOCAL_B_Q_I: for (ii=0; ii<NUMBER_PAR_PER_BOX; ++ii){
#pragma HLS PIPELINE
                local_B_q_i[ii] = q_i[B_idx+ii];
            }
            //calculate the accumulated effects from all neighboring particles
            PARTICLES_B: for(k=0; k<NUMBER_PAR_PER_BOX; ++k){
                PARTICLES_A: for(j=0; j<NUMBER_PAR_PER_BOX; j+=UNROLL_SIZE){
#pragma HLS PIPELINE II=2
                    UNROLLING_LOOP: for(jj=0; jj<UNROLL_SIZE; ++jj){
#pragma HLS UNROLL
#pragma HLS DEPENDENCE variable="local_pos_o" inter false
                        // coefficients
                        r2[jj] = local_A_pos_i[j+jj][V] + local_B_pos_i[k][V] -
                            (local_A_pos_i[j+jj][X] * local_B_pos_i[k][X] +
                             local_A_pos_i[j+jj][Y] * local_B_pos_i[k][Y] +
                             local_A_pos_i[j+jj][Z] * local_B_pos_i[k][Z]); //DOT(pos_i[A_idx_j], pos_i[B_idx_k]);
                        u2[jj] = A2 * r2[jj];
                        //Below line equivalent as -> vij = exp(-u2);
                        u2[jj] = u2[jj] * -1.0;
                        vij[jj] = 1.0 + (u2[jj]) + 0.5*((u2[jj])*(u2[jj])) +
                                0.16666*((u2[jj])*(u2[jj])*(u2[jj])) +
                                0.041666*((u2[jj])*(u2[jj])*(u2[jj])*(u2[jj]));
                        fs[jj] = 2. * vij[jj];
                        d[jj][X] = local_A_pos_i[j+jj][X] - local_B_pos_i[k][X];
                        d[jj][Y] = local_A_pos_i[j+jj][Y] - local_B_pos_i[k][Y];
                        d[jj][Z] = local_A_pos_i[j+jj][Z] - local_B_pos_i[k][Z];
                        fxij[jj] = fs[jj] * d[jj][X];
                        fyij[jj] = fs[jj] * d[jj][Y];
                        fzij[jj] = fs[jj] * d[jj][Z];
                        // forces
                        local_pos_o[j+jj][V] += local_B_q_i[k] * vij[jj];
                        local_pos_o[j+jj][X] += local_B_q_i[k] * fxij[jj];
                        local_pos_o[j+jj][Y] += local_B_q_i[k] * fyij[jj];
                        local_pos_o[j+jj][Z] += local_B_q_i[k] * fzij[jj];
                    }
                }
            }
        }
        //writeback local_pos_o
        WRITE_POS_O: for (ii=0; ii<NUMBER_PAR_PER_BOX; ++ii){
#pragma HLS PIPELINE II=2
            pos_o[(C_idx+ii)*POS_DIM+V] = local_pos_o[ii][V];
            pos_o[(C_idx+ii)*POS_DIM+X] = local_pos_o[ii][X];
            pos_o[(C_idx+ii)*POS_DIM+Y] = local_pos_o[ii][Y];
            pos_o[(C_idx+ii)*POS_DIM+Z] = local_pos_o[ii][Z];
        }
    }
}

extern "C"
{
    void workload(TYPE pos_i[N_PADDED*POS_DIM], TYPE q_i[N_PADDED], TYPE pos_o[N*POS_DIM])
    {
        #pragma HLS INTERFACE m_axi port=pos_i  offset=slave bundle=gmem
        #pragma HLS INTERFACE m_axi port=q_i    offset=slave bundle=gmem
        #pragma HLS INTERFACE m_axi port=pos_o  offset=slave bundle=gmem

        #pragma HLS INTERFACE s_axilite port=pos_i  bundle=control
        #pragma HLS INTERFACE s_axilite port=q_i    bundle=control
        #pragma HLS INTERFACE s_axilite port=pos_o  bundle=control

        #pragma HLS INTERFACE s_axilite port=return bundle=control

        lavaMD_unrolling_padded(pos_i, q_i, pos_o);

        return;
    }
}
