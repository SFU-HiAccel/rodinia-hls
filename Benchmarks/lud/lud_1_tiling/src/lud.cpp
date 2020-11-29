#include"lud.h"
#include <iostream>

//Elment with the block BSIZE, diagonal
#define AA(i,j) result[(offset + i) * matrix_dim + j + offset]
//Elment with global index
#define BB(i,j) result[i * matrix_dim + j]

using namespace std;
extern "C"{
void diagonal_load(float* result, float* buffer, int offset){
    int i, j;
    for(i = 0; i < BSIZE; i++){
        for (j = 0; j < BSIZE; j++){
            buffer[i * BSIZE + j] = AA(i, j);
        }
    }
}

void diagonal_store(float* result, float* buffer, int offset){
    int i, j;
    for(i = 0; i < BSIZE; i++){
        for (j = 0; j < BSIZE; j++){
            AA(i, j) = buffer[i * BSIZE + j];
        }
    }
}
void lud_diagonal(float* result, 
                  int offset)
{ 
    int i, j, k;

    float buffer[BSIZE * BSIZE];

    diagonal_load(result, buffer, offset);

    for (i = 0; i < BSIZE; i++){
        top:for (j = i; j < BSIZE; j++){
            for (k = 0; k < i; k++){
                buffer[i * BSIZE + j] = buffer[i * BSIZE + j] - 
                buffer[i * BSIZE + k]* buffer[k * BSIZE + j];
            }
        }
           
           float temp = 1.f / buffer[i * BSIZE + i];

           left:for (j = i + 1; j < BSIZE; j++){
               for (k = 0; k < i; k++){
                   buffer[j * BSIZE + i]= buffer[j * BSIZE + i]- buffer[j * BSIZE + k] * buffer[k * BSIZE + i];
               }
               buffer[j * BSIZE + i] = buffer[j * BSIZE + i] * temp;
           }
    }

    diagonal_store(result, buffer, offset);
}

void perimeter_load(float* result, float* top, float* left, int offset, int chunk_idx){
    int i, j;

    int i_top = offset; 
    int j_top = offset + BSIZE * (chunk_idx + 1);

    int i_left = offset + BSIZE * (chunk_idx + 1);
    int j_left = offset;

    for(i = 0; i < BSIZE; i++){
        for(j = 0; j < BSIZE; j++){
            top[i * BSIZE + j] = BB((i_top + i), (j_top + j));
        }
    }

    for(i = 0; i < BSIZE; i++){
        for(j = 0; j < BSIZE; j++){
            left[i * BSIZE + j] = BB((i_left + i), (j_left + j));
        }
    }
}

void perimeter_store(float* result, float* top, float* left, int offset, int chunk_idx){
    int i, j;

    int i_top = offset; 
    int j_top = offset + BSIZE * (chunk_idx + 1);

    int i_left = offset + BSIZE * (chunk_idx + 1);
    int j_left = offset;

    for(i = 0; i < BSIZE; i++){
        for(j = 0; j < BSIZE; j++){
            BB((i_top + i), (j_top + j)) = top[i * BSIZE + j];
        }
    }

    for(i = 0; i < BSIZE; i++){
        for(j = 0; j < BSIZE; j++){
            BB((i_left + i), (j_left + j)) = left[i * BSIZE + j];
        }
    }
}
//99327
void lud_perimeter(float* result, 
                   int    offset) {
    float diagonal_buffer[BSIZE * BSIZE];
    float top_buffer[BSIZE * BSIZE];
    float left_buffer[BSIZE * BSIZE];

    int i, j, k;

    diagonal:for (i = 0; i < BSIZE; i++){
        for (j = 0; j < BSIZE; j++){
            diagonal_buffer[i * BSIZE + j] = AA(i, j);
        }
    }

    int chunk_idx, chunk_num;

    chunk_num = ((matrix_dim - offset) / BSIZE) - 1;

    for (chunk_idx = 0; chunk_idx < chunk_num; chunk_idx++){
        perimeter_load(result, top_buffer, left_buffer, offset, chunk_idx);
                
                        float sum; 
                        // processing top perimeter
                        for (j = 0; j < BSIZE; j++){
                            for (i = 0; i < BSIZE; i++){ 
                                sum = 0.0f; 
                                for (k = 0; k < i; k++){
                                    sum += diagonal_buffer[BSIZE * i + k] * top_buffer[k * BSIZE + j];
                                }

                                top_buffer[i * BSIZE + j] = top_buffer[i * BSIZE + j] - sum;
                            }
                        }


                        // processing left perimeter
                        for (i = 0; i < BSIZE; i++){
                            for (j = 0; j < BSIZE; j++){
                                 sum = 0.0f;
                                 for (k = 0; k < j; k++){
                                     sum += left_buffer[i * BSIZE + k] * diagonal_buffer[BSIZE * k + j];
                                 }

                                 left_buffer[i * BSIZE + j] = (left_buffer[i * BSIZE + j] - sum) / diagonal_buffer[j * BSIZE + j];
                            }
                        }

                        perimeter_store(result, top_buffer, left_buffer, offset, chunk_idx);

    }
    cout << "success here perimeter" << endl;
}

void internal_load(float* result, float* top, float* left, float* inner, int offset, int chunk_idx, int chunk_num){
    int i, j;
    int i_global, j_global;

    i_global = offset + BSIZE * (1 + chunk_idx / chunk_num);
    j_global = offset + BSIZE * (1 + chunk_idx % chunk_num);

    for(i = 0; i < BSIZE; i++){
        for(j = 0; j < BSIZE; j++){
            top[i * BSIZE + j]  = result[matrix_dim * (i + offset) + j + j_global];
        }
    }
    for(i = 0; i < BSIZE; i++){
        for(j = 0; j < BSIZE; j++){
            left[i * BSIZE + j] = result[matrix_dim * (i + i_global) + offset + j];
        }
    }

    for(i = 0; i < BSIZE; i++){
        for(j = 0; j < BSIZE; j++){
            inner[i * BSIZE + j] = result[matrix_dim * (i + i_global) + j + j_global];
        }
    }
}

void internal_store(float* result, float* inner, int offset, int chunk_idx, int chunk_num){
    int i, j;
    int i_global, j_global;

    i_global = offset + BSIZE * (1 + chunk_idx / chunk_num);
    j_global = offset + BSIZE * (1 + chunk_idx % chunk_num);

    for(i = 0; i < BSIZE; i++){
        for(j = 0; j < BSIZE; j++){
            result[matrix_dim * (i + i_global) + j + j_global] = inner[i * BSIZE + j];
        }
    }
}

void lud_internal( float* result, 
                    int  offset) {
    int chunk_idx, chunk_num;

    chunk_num = ((matrix_dim - offset) / BSIZE) - 1;

    float top_buffer[BSIZE * BSIZE];
    float left_buffer[BSIZE * BSIZE];
    float inner_buffer[BSIZE * BSIZE];

    int i, j, k, i_global, j_global;

    for  (chunk_idx = 0; chunk_idx < chunk_num * chunk_num; chunk_idx++){
        internal_load(result, top_buffer, left_buffer, inner_buffer, offset, chunk_idx, chunk_num);

        for (i = 0; i < BSIZE; i++){
            for (j = 0; j < BSIZE; j++){
                float sum = 0.0f;
                //#pragma HLS unsafemath
                for (k = 0; k < BSIZE; k++){
                    sum += left_buffer[BSIZE * i + k] * top_buffer[BSIZE * k + j];
                }
                inner_buffer[i * BSIZE + j] -= sum;
            }
        }

        internal_store(result, inner_buffer, offset, chunk_idx, chunk_num);
    }

    cout << "success internal" << endl;
}

void workload(float result[GRID_ROWS * GRID_COLS]){
#pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=result bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    for(int i = 0; i < matrix_dim - BSIZE; i += BSIZE){
        lud_diagonal(result, i);
        lud_perimeter(result, i);
        lud_internal(result, i);
    }

    int i = matrix_dim - BSIZE;
    lud_diagonal(result, i);
    return;
}

}
