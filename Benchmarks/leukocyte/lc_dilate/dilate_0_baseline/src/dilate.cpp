#include "dilate.h"

extern "C" {

void workload(float result[GRID_ROWS * GRID_COLS], float img[(GRID_ROWS+2*MAX_RADIUS)*GRID_COLS])
{

	#pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem
	#pragma HLS INTERFACE m_axi port=img offset=slave bundle=gmem
	#pragma HLS INTERFACE s_axilite port=result bundle=control
	#pragma HLS INTERFACE s_axilite port=img bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	int starting_idx = MAX_RADIUS*GRID_COLS;

	bool strel[25] = { 0, 0, 1, 0, 0,
					   0, 1, 1, 1, 0,
					   1, 1, 1, 1, 1,
					   0, 1, 1, 1, 0,
					   0, 0, 1, 0, 0 };

	int radius_p = STREL_ROWS / 2;
	int radius_q = STREL_COLS / 2;

	for (int i = 0; i < GRID_ROWS; i++) {
		for (int j = 0; j < GRID_COLS; j++) {
			float max = 0.0, temp;
			for (int m = 0; m < STREL_ROWS; m++) {
				for (int n = 0; n < STREL_COLS; n++) {
					int p = i - radius_p + m;
					int q = j - radius_q + n;
					if (p >= 0 && q >= 0 && p < GRID_ROWS && q < GRID_COLS && strel[m * STREL_COLS + n] != 0) {
						temp = img[starting_idx + p * GRID_COLS + q];
						if (temp > max) max = temp;
					}
				}
			}
			result[i * GRID_COLS + j] = max;
		}
	}

	return;
}

}