#include "lavaMD.h"

void lavaMD_tiling_padded(TYPE pos_i[N_PADDED*POS_DIM], TYPE q_i[N_PADDED], TYPE pos_o[N*POS_DIM])
{
    //local variables
    int i, j, k;
    int ii, jj, kk;
    int x, y, z;
    int l, m, n;
    int x_n, y_n, z_n;
    int A_idx, B_idx, C_idx;
    int remainder;
    TYPE r2, u2, fs, vij, fxij, fyij, fzij;
    TYPE d[POS_DIM];

    int neighborOffset[FULL_NEIGHBOR_COUNT][3] = {{-1,-1,-1}, { 0,-1,-1}, { 1,-1,-1},
                                                  {-1, 0,-1}, { 0, 0,-1}, { 1, 0,-1},
                                                  {-1, 1,-1}, { 0, 1,-1}, { 1, 1,-1},
                                                  {-1,-1, 0}, { 0,-1, 0}, { 1,-1, 0},
                                                  {-1, 0, 0}, { 0, 0, 0}, { 1, 0, 0},
                                                  {-1, 1, 0}, { 0, 1, 0}, { 1, 1, 0},
                                                  {-1,-1, 1}, { 0,-1, 1}, { 1,-1, 1},
                                                  {-1, 0, 1}, { 0, 0, 1}, { 1, 0, 1},
                                                  {-1, 1, 1}, { 0, 1, 1}, { 1, 1, 1}};
    //local BRAM - tiling essentials
    TYPE local_A_pos_i[NUMBER_PAR_PER_BOX*POS_DIM];
    TYPE local_B_pos_i[NUMBER_PAR_PER_BOX*POS_DIM];
    TYPE local_B_q_i[NUMBER_PAR_PER_BOX];
    TYPE local_pos_o[NUMBER_PAR_PER_BOX*POS_DIM];

    //iterate through all boxes in the 3D space
    BOXES: for(i=0; i<DIMENSION_3D; i++){
        //purge local_pos_o
        WIPE_LOCAL_POS_O: for (ii=0; ii<NUMBER_PAR_PER_BOX; ++ii){
            local_pos_o[ii*POS_DIM+V] = 0.0;
            local_pos_o[ii*POS_DIM+X] = 0.0;
            local_pos_o[ii*POS_DIM+Y] = 0.0;
            local_pos_o[ii*POS_DIM+Z] = 0.0;
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
            local_A_pos_i[ii*POS_DIM+V] = pos_i[(A_idx+ii)*POS_DIM+V];
            local_A_pos_i[ii*POS_DIM+X] = pos_i[(A_idx+ii)*POS_DIM+X];
            local_A_pos_i[ii*POS_DIM+Y] = pos_i[(A_idx+ii)*POS_DIM+Y];
            local_A_pos_i[ii*POS_DIM+Z] = pos_i[(A_idx+ii)*POS_DIM+Z];
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
                local_B_pos_i[ii*POS_DIM+V] = pos_i[(B_idx+ii)*POS_DIM+V];
                local_B_pos_i[ii*POS_DIM+X] = pos_i[(B_idx+ii)*POS_DIM+X];
                local_B_pos_i[ii*POS_DIM+Y] = pos_i[(B_idx+ii)*POS_DIM+Y];
                local_B_pos_i[ii*POS_DIM+Z] = pos_i[(B_idx+ii)*POS_DIM+Z];
            }
            //load local_B_q_i
            LOAD_LOCAL_B_Q_I: for (ii=0; ii<NUMBER_PAR_PER_BOX; ++ii){
                local_B_q_i[ii] = q_i[B_idx+ii];
            }

            //calculate the accumulated effects from all neighboring particles
            PARTICLES_A: for(j=0; j<NUMBER_PAR_PER_BOX; ++j){
                PARTICLES_B: for(k=0; k<NUMBER_PAR_PER_BOX; ++k){
                    // coefficients
                    r2 = local_A_pos_i[j*POS_DIM+V] + local_B_pos_i[k*POS_DIM+V] -
                        (local_A_pos_i[j*POS_DIM+X] * local_B_pos_i[k*POS_DIM+X] +
                         local_A_pos_i[j*POS_DIM+Y] * local_B_pos_i[k*POS_DIM+Y] +
                         local_A_pos_i[j*POS_DIM+Z] * local_B_pos_i[k*POS_DIM+Z]); 
                    u2 = A2 * r2;
                    //Below line equivalent as -> vij = exp(-u2);
                    u2 = u2 * -1.0;
                    vij = 1.0 + (u2) + 0.5*((u2)*(u2)) +
                            0.16666*((u2)*(u2)*(u2)) +
                            0.041666*((u2)*(u2)*(u2)*(u2));
                    fs = 2. * vij;
                    d[X] = local_A_pos_i[j*POS_DIM+X] - local_B_pos_i[k*POS_DIM+X];
                    d[Y] = local_A_pos_i[j*POS_DIM+Y] - local_B_pos_i[k*POS_DIM+Y];
                    d[Z] = local_A_pos_i[j*POS_DIM+Z] - local_B_pos_i[k*POS_DIM+Z];
                    fxij = fs * d[X];
                    fyij = fs * d[Y];
                    fzij = fs * d[Z];
                    // forces
                    local_pos_o[j*POS_DIM+V] += local_B_q_i[k] * vij;
                    local_pos_o[j*POS_DIM+X] += local_B_q_i[k] * fxij;
                    local_pos_o[j*POS_DIM+Y] += local_B_q_i[k] * fyij;
                    local_pos_o[j*POS_DIM+Z] += local_B_q_i[k] * fzij;                    
                }
            }
        }
        //writeback local_pos_o
        WRITE_POS_O: for (ii=0; ii<NUMBER_PAR_PER_BOX; ++ii){
            pos_o[(C_idx+ii)*POS_DIM+V] = local_pos_o[ii*POS_DIM+V];
            pos_o[(C_idx+ii)*POS_DIM+X] = local_pos_o[ii*POS_DIM+X];
            pos_o[(C_idx+ii)*POS_DIM+Y] = local_pos_o[ii*POS_DIM+Y];
            pos_o[(C_idx+ii)*POS_DIM+Z] = local_pos_o[ii*POS_DIM+Z];
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

        lavaMD_tiling_padded(pos_i, q_i, pos_o);

        return;
    }
}
