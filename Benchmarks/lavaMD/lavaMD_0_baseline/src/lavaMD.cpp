#include "lavaMD.h"

void lavaMD_baseline_padded(TYPE pos_i[N_PADDED*POS_DIM], TYPE q_i[N_PADDED], TYPE pos_o[N*POS_DIM])
{
    int i, j, k;
    int x, y, z;
    int l, m, n;
    int x_n, y_n, z_n;
    int A_idx, B_idx, C_idx, A_idx_j, B_idx_k, C_idx_j;
    int remainder;

    TYPE r2;
    TYPE u2;
    TYPE fs;
    TYPE vij;
    TYPE fxij, fyij, fzij;
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
    //iterate through all boxes in the 3D space
    BOXES: for(i=0; i<DIMENSION_3D; i++){
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

        for (l=0; l<FULL_NEIGHBOR_COUNT; ++l){
            //pos of the neighbor - input_B array
            x = x_n + neighborOffset[l][0];
            y = y_n + neighborOffset[l][1];
            z = z_n + neighborOffset[l][2];
            //convert from 3D-index to 1D-index
            B_idx = z*DIMENSION_2D_PADDED + y*DIMENSION_1D_PADDED + x;
            B_idx = B_idx * NUMBER_PAR_PER_BOX;

            //calculate the accumulated effects from all neighboring particles
            PARTICLES_A: for(j=0; j<NUMBER_PAR_PER_BOX; ++j){
                A_idx_j = A_idx + j;
                C_idx_j = C_idx + j;
                PARTICLES_B: for(k=0; k<NUMBER_PAR_PER_BOX; ++k){
                    B_idx_k = B_idx + k;
                    // coefficients - DOT(pos_i[A_idx_j], pos_i[B_idx_k]);
                    r2 = pos_i[A_idx_j*POS_DIM+V] + pos_i[B_idx_k*POS_DIM+V] -
                        (pos_i[A_idx_j*POS_DIM+X] * pos_i[B_idx_k*POS_DIM+X] +
                         pos_i[A_idx_j*POS_DIM+Y] * pos_i[B_idx_k*POS_DIM+Y] +
                         pos_i[A_idx_j*POS_DIM+Z] * pos_i[B_idx_k*POS_DIM+Z]);
                    u2 = A2 * r2;
                    //Below line equivalent as -> vij = exp(-u2);
                    u2 = u2 * -1.0;
                    vij = 1.0 + (u2) + 0.5*((u2)*(u2)) +
                            0.16666*((u2)*(u2)*(u2)) +
                            0.041666*((u2)*(u2)*(u2)*(u2));
                    fs = 2. * vij;
                    d[X] = pos_i[A_idx_j*POS_DIM+X] - pos_i[B_idx_k*POS_DIM+X];
                    d[Y] = pos_i[A_idx_j*POS_DIM+Y] - pos_i[B_idx_k*POS_DIM+Y];
                    d[Z] = pos_i[A_idx_j*POS_DIM+Z] - pos_i[B_idx_k*POS_DIM+Z];
                    fxij = fs * d[X];
                    fyij = fs * d[Y];
                    fzij = fs * d[Z];
                    // forces
                    pos_o[C_idx_j*POS_DIM+V] += q_i[B_idx_k] * vij;
                    pos_o[C_idx_j*POS_DIM+X] += q_i[B_idx_k] * fxij;
                    pos_o[C_idx_j*POS_DIM+Y] += q_i[B_idx_k] * fyij;
                    pos_o[C_idx_j*POS_DIM+Z] += q_i[B_idx_k] * fzij;
                }
            }
        }
    }
}

extern "C"
{
    void workload(TYPE pos_i[N_PADDED*POS_DIM], TYPE q_i[N_PADDED], TYPE pos_o[N*POS_DIM])
    {
        // #pragma HLS INTERFACE m_axi port=pos_i  offset=slave bundle=gmem0
        // #pragma HLS INTERFACE m_axi port=q_i    offset=slave bundle=gmem1
        // #pragma HLS INTERFACE m_axi port=pos_o  offset=slave bundle=gmem2
        #pragma HLS INTERFACE m_axi port=pos_i  offset=slave bundle=gmem
        #pragma HLS INTERFACE m_axi port=q_i    offset=slave bundle=gmem
        #pragma HLS INTERFACE m_axi port=pos_o  offset=slave bundle=gmem

        #pragma HLS INTERFACE s_axilite port=pos_i  bundle=control
        #pragma HLS INTERFACE s_axilite port=q_i    bundle=control
        #pragma HLS INTERFACE s_axilite port=pos_o  bundle=control

        #pragma HLS INTERFACE s_axilite port=return bundle=control

        lavaMD_baseline_padded(pos_i, q_i, pos_o);

        return;
    }
}
