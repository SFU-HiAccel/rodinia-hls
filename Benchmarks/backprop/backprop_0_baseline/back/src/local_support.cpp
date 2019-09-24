#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include "backprop.h"
#include <CL/opencl.h>
#include "my_timer.h"

int INPUT_SIZE = 1;

extern void setup(cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel);

void bpnn_adjust_weights_FPGA(float *hidden, float *input, float *weight, float *prev_weight, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel) {
  cl_mem d_hidden;
  cl_mem d_input;
  cl_mem d_weight;
  cl_mem d_prev_weight;

  cl_int err = 0;


  // 0th: initialize the timer at the beginning of the program
  timespec timer = tic();

  // Create device buffers
  d_hidden = clCreateBuffer(context, CL_MEM_READ_WRITE, 17 * sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_hidden (size:17) => %d\n", err); }
  d_input = clCreateBuffer(context, CL_MEM_READ_WRITE, 65537 * sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_input (size:65537) => %d\n", err); }
  d_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, 65537 * 17 * sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_weight (size:65537*17) => %d\n", err);}
    d_prev_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, 65537 * 17 * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_prev_weight (size:65537*17) => %d\n", err); }

  // 1st: time of buffer allocation
  toc(&timer, "buffer allocation");

  // Write our data set into device buffers
  //

  err = clEnqueueWriteBuffer(commands, d_hidden, 1, 0, 17 * sizeof(float), hidden, 0, 0, 0);
  if (err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_hidden (size:17) => %d\n", err); }
  err = clEnqueueWriteBuffer(commands, d_input, 1, 0, 65537 * sizeof(float), input, 0, 0, 0);
  if (err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_input (size:65537) => %d\n", err); }
  err = clEnqueueWriteBuffer(commands, d_weight, 1, 0, 65537 * 17 * sizeof(float), weight, 0, 0, 0);
  if (err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_weight (size:65537*17) => %d\n", err); }
    err = clEnqueueWriteBuffer(commands, d_prev_weight, 1, 0, 65537 * 17 * sizeof(float), prev_weight, 0, 0, 0);
    if (err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_prev_weight (size:65537*17) => %d\n", err);}

  // 2nd: time of pageable-pinned memory copy
  toc(&timer, "memory copy");

  // Set the arguments to our compute kernel
  err = clSetKernelArg(kernel, 0, sizeof(void *), (void *) &d_hidden);
  err |= clSetKernelArg(kernel, 1, sizeof(void *), (void *) &d_input);
  err |= clSetKernelArg(kernel, 2, sizeof(void *), (void *) &d_weight);
    err |= clSetKernelArg(kernel, 3, sizeof(void *), (void *) &d_prev_weight);

  if (err != CL_SUCCESS) {
    printf("Error: Failed to set kernel arguments! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }

  // 3rd: time of setting arguments
  toc(&timer, "set arguments");

  // Execute the kernel over the entire range of our 1d input data set
  // using the maximum number of work group items for this device
  //


  err = clEnqueueTask(commands, kernel, 0, NULL, NULL);
  if (err) {
    printf("Error: Failed to execute kernel! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }

  clFinish(commands);

  // 4th: time of kernel execution
  toc(&timer, "kernel execution");

  err = clEnqueueReadBuffer(commands, d_weight, 1, 0, 65537 * 17 * sizeof(float), weight, 0, 0, 0);
  if (err != CL_SUCCESS) { printf("ERROR: Memcopy Out\n"); }

    err = clEnqueueReadBuffer(commands, d_prev_weight, 1, 0, 65537 * 17 * sizeof(float), prev_weight, 0, 0, 0);
    if (err != CL_SUCCESS) { printf("ERROR: Memcopy Out\n"); }

  // 5th: time of data retrieving (PCIe + memcpy)
  toc(&timer, "data retrieving");

  clReleaseMemObject(d_input);
  clReleaseMemObject(d_hidden);
  clReleaseMemObject(d_weight);
  clReleaseMemObject(d_prev_weight);
}

void run_benchmark(void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  setup(context, commands, program, kernel);
}

void input_to_data(int fd, void *vdata) {

}

void data_to_input(int fd, void *vdata) {

}

void output_to_data(int fd, void *vdata) {

}

void data_to_output(int fd, void *vdata) {

}

int check_data( void *vdata, void *vref ) {
  // Return true if it's correct.
  return 1;
}