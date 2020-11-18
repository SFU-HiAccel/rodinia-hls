#include "nw.h"
#include "support.h"
#include <string.h>
#include "my_timer.h"

int INPUT_SIZE = sizeof(struct bench_args_t);

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;
  int num_jobs = 1 << 10;

// Create host buffer
  char* seqA_batch = (char *)malloc(sizeof(args->seqA) * num_jobs);
  char* seqB_batch = (char *)malloc(sizeof(args->seqB) * num_jobs);
  char* alignedA_batch = (char *)malloc(sizeof(args->alignedA) * num_jobs);
  char* alignedB_batch = (char *)malloc(sizeof(args->alignedB) * num_jobs);
  
  int i;
  for (i=0; i<num_jobs; i++) {
    memcpy(seqA_batch + i*sizeof(args->seqA), args->seqA, sizeof(args->seqA));
    memcpy(seqB_batch + i*sizeof(args->seqB), args->seqB, sizeof(args->seqB));
  }

  printf("sizeof(seqA_batch): %d \n", sizeof(seqA_batch));
  printf("sizeof(args->seqA): %d \n", sizeof(args->seqA));
  printf("\n");

  // 0th: initialize the timer at the beginning of the program
  timespec timer = tic();

  // Create device buffers
  //
  cl_mem seqA_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->seqA)*num_jobs, NULL, NULL);
  cl_mem seqB_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->seqB)*num_jobs, NULL, NULL);
  cl_mem alignedA_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->alignedA)*num_jobs, NULL, NULL);
  cl_mem alignedB_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->alignedB)*num_jobs, NULL, NULL);
  if (!seqA_buffer || !seqB_buffer || !alignedA_buffer || !alignedB_buffer)
  {
    printf("Error: Failed to allocate device memory!\n");
    printf("Test failed\n");
    exit(1);
  }

  // 1st: time of buffer allocation
  toc(&timer, "buffer allocation");

  // Write our data set into device buffers
  //
  int err;
  err = clEnqueueWriteBuffer(commands, seqA_buffer, CL_TRUE, 0, sizeof(args->seqA)*num_jobs, seqA_batch, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, seqB_buffer, CL_TRUE, 0, sizeof(args->seqB)*num_jobs, seqB_batch, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
      printf("Error: Failed to write to device memory!\n");
      printf("Test failed\n");
      exit(1);
  }

  // 2nd: time of pageable-pinned memory copy
  toc(&timer, "memory copy");

  // Set the arguments to our compute kernel
  //
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &seqA_buffer);
  err  |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &seqB_buffer);
  err  |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &alignedA_buffer);
  err  |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &alignedB_buffer);
  err  |= clSetKernelArg(kernel, 4, sizeof(int), &num_jobs);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to set kernel arguments! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }

  // 3rd: time of setting arguments
  toc(&timer, "set arguments");

  // Execute the kernel over the entire range of our 1d input data set
  // using the maximum number of work group items for this device
  //

#ifdef C_KERNEL
  err = clEnqueueTask(commands, kernel, 0, NULL, NULL);
#else
  printf("Error: OpenCL kernel is not currently supported!\n");
  exit(1);
#endif
  if (err)
  {
    printf("Error: Failed to execute kernel! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }

  // 4th: time of kernel execution
  clFinish(commands);
  toc(&timer, "kernel execution");

  // Read back the results from the device to verify the output
  //
  err = clEnqueueReadBuffer( commands, alignedA_buffer, CL_TRUE, 0, sizeof(args->alignedA)*num_jobs, alignedA_batch, 0, NULL, NULL );
  err |= clEnqueueReadBuffer( commands, alignedB_buffer, CL_TRUE, 0, sizeof(args->alignedB)*num_jobs, alignedB_batch, 0, NULL, NULL );
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }

  // 5th: time of data retrieving (PCIe + memcpy)
  toc(&timer, "data retrieving");

  memcpy(args->alignedA, alignedA_batch, sizeof(args->alignedA));
  memcpy(args->alignedB, alignedB_batch, sizeof(args->alignedB));
  free(seqA_batch);
  free(seqB_batch);
  free(alignedA_batch);
  free(alignedB_batch);
}

/* Input format:
%% Section 1
char[]: sequence A
%% Section 2
char[]: sequence B
*/

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  parse_string(s, data->seqA, ALEN);

  s = find_section_start(p,2);
  parse_string(s, data->seqB, BLEN);

}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  write_string(fd, data->seqA, ALEN);

  write_section_header(fd);
  write_string(fd, data->seqB, BLEN);

  write_section_header(fd);
}

/* Output format:
%% Section 1
char[sum_size]: aligned sequence A
%% Section 2
char[sum_size]: aligned sequence B
*/

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  parse_string(s, data->alignedA, ALEN+BLEN);

  s = find_section_start(p,2);
  parse_string(s, data->alignedB, ALEN+BLEN);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  write_string(fd, data->alignedA, ALEN+BLEN);

  write_section_header(fd);
  write_string(fd, data->alignedB, ALEN+BLEN);

  write_section_header(fd);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;

  has_errors |= memcmp(data->alignedA, ref->alignedA, ALEN+BLEN);
  has_errors |= memcmp(data->alignedB, ref->alignedB, ALEN+BLEN);

  // Return true if it's correct.
  return !has_errors;
}
