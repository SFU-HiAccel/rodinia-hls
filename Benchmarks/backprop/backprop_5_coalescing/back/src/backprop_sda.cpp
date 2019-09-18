#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "backprop.h"
#include <CL/opencl.h>

extern void bpnn_adjust_weights_FPGA(float *hidden, float *input, float *weight, float *prev_weight, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel);
extern void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);

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
    float out_err, hid_err;

    in = net->input_n;
    hid = net->hidden_n;
    out = net->output_n;

    bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in, hid);
    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
    bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

    float *input_weights_FPGA = (float *)malloc(65537*17*sizeof(float));
    float *input_prev_weights_FPGA = (float *)malloc(65537*17*sizeof(float));
    for (int j = 0; j <= 16; j++) {
        for (int k = 0; k <= 65536; k++) {
            input_weights_FPGA[k*17+j] = net->input_weights[k][j];
        }
    }

    for (int i = 0; i < 65537; i++) {
        for (int j = 0; j < 17; j++) {
            input_prev_weights_FPGA[i*17+j] = 0.0;
        }
    }

    bpnn_adjust_weights_FPGA(net->hidden_delta, net->input_units, input_weights_FPGA, input_prev_weights_FPGA, context, commands, program, kernel);
    bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

    int error = 0;
    for (int j = 1; j <= 16; j++) {
        for (int k = 0; k <= 65536; k++) {
            if (rcmp(input_weights_FPGA[k*17+j], net->input_weights[k][j]) > 1e-12) {
                error++;
            }

            if (rcmp(input_prev_weights_FPGA[k*17+j], net->input_prev_weights[k][j]) > 1e-12) {
                error++;
            }
        }
    }

    printf("Error %d\n", error);
	free(input_weights_FPGA);
	free(input_prev_weights_FPGA);
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
