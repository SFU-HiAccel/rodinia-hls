#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "backprop.h"
#include <CL/opencl.h>

extern void bpnn_layerforward_FPGA(float *input, float *hidden, float *conn, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel);

extern void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern void bpnn_save(BPNN *net, char *filename);

extern void bpnn_initialize(int seed);

extern BPNN *bpnn_create(int n_in, int n_hidden, int n_out);

extern void bpnn_free(BPNN *net);

extern float **alloc_2d_dbl(int m, int n);
////////////////////////////////////////////////////////////////////////////////


double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

void load(BPNN *net) {
    float *units;
    int nr, i, k;

    nr = 65536;
    units = net->input_units;

    k = 1;
    for (i = 0; i < nr; i++) {
        units[k] = (float) rand() / RAND_MAX;
        k++;
    }
}

float rcmp(float a, float b) {
    return fabs((a - b) / (a + b));
}

void bpnn_train_kernel(BPNN *net, float *eo, float *eh, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel) {
    int in, hid, out;

    in = net->input_n;
    hid = net->hidden_n;
    out = net->output_n;

    float hidden_units_FPGA[17];
	float *input_weights_FPGA = (float *)malloc(65537*16*sizeof(float));
    for (int j = 1; j <= 16; j++) {
        for (int k = 0; k <= 65536; k++) {
            input_weights_FPGA[k*16+j-1] = net->input_weights[k][j];
        }
    }

    bpnn_layerforward_FPGA(net->input_units, hidden_units_FPGA, input_weights_FPGA, context, commands, program, kernel);

    bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in, hid);

    int error = 0;
    for(int i = 1; i <= 16; i++) {
        printf("%f %f\n", hidden_units_FPGA[i], net->hidden_units[i]);
        if(rcmp(hidden_units_FPGA[i], net->hidden_units[i]) > 1e-4) {
            error++;
        }
    }

    printf("Error %d\n", error);
	free(input_weights_FPGA);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
void setup(cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel) {
    int seed;

    seed = 7;
    bpnn_initialize(seed);

    BPNN *net;

    float out_err, hid_err;
    net = bpnn_create(65536, 16, 1); // (16, 1 can not be changed)
    load(net);
    //entering the training kernel, only one iteration
    printf("Starting training kernel\n");
    bpnn_train_kernel(net, &out_err, &hid_err, context, commands, program, kernel);
    bpnn_free(net);
    printf("Training done\n");

    exit(0);
}
