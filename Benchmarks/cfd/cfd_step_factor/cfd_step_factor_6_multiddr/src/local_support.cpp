#include <string.h>
#include <CL/opencl.h>
#include "my_timer.h"
#include "support.h"
#include "cfd_step_factor.h"

int INPUT_SIZE = sizeof(struct bench_args_t);

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;

  // 0th: initialize the timer at the beginning of the program
  timespec timer = tic();
    
    cl_mem result_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args -> result), NULL, NULL);
    cl_mem variables_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(args -> variables), NULL, NULL);
    cl_mem areas_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(args -> areas), NULL, NULL);

  if (!result_buffer || !variables_buffer || !areas_buffer)
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
    err  = clEnqueueWriteBuffer(commands, result_buffer  , CL_TRUE, 0, sizeof(args -> result), args -> result  , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, variables_buffer  , CL_TRUE, 0, sizeof(args -> variables), args -> variables  , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, areas_buffer  , CL_TRUE, 0, sizeof(args -> areas), args -> areas  , 0, NULL, NULL);

  if (err != CL_SUCCESS)
  {
      printf("Error: Failed to write to device memory!\n");
      printf("Test failed\n");
      exit(1);
  }

  // 2nd: time of pageable-pinned memory copy
  toc(&timer, "memory copy");
    
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &result_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &variables_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &areas_buffer);

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
  
    err  = clEnqueueReadBuffer(commands, result_buffer,  CL_TRUE, 0, sizeof(args -> result)  , args -> result  , 0, NULL, NULL);  


  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    exit(1);
 }

  // 5th: time of data retrieving (PCIe + memcpy)
  toc(&timer, "data retrieving");

}

/* Input format:
%% Section 1
char[]: variables
%% Section 2
char[]: areas
*/

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
int i;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);
  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data -> variables, SIZE * NVAR);

  s = find_section_start(p,2);
  STAC(parse_,TYPE,_array)(s, data -> areas, SIZE);
//printf("%d FD:++++++++++++++++++++++++++++", fd);

for (i = 0; i < SIZE; i++) {
    data -> result[i] = 0;
}

}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_, TYPE, _array)(fd, data -> result, SIZE);
  
  write_section_header(fd);
  STAC(write_, TYPE, _array)(fd, data -> variables, SIZE * NVAR);

  write_section_header(fd);
  STAC(write_, TYPE, _array)(fd, data -> areas, SIZE);

  write_section_header(fd);
}


void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->result, SIZE);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  FILE* fid = fopen("output.data", "w");

  for (int kk = 0; kk <SIZE;kk++){
    fprintf(fid,"%.18f\n", data->result[kk]);
  }

  fclose(fid);

  printf("+++++++++++++++++++++++++++++++++++data_to_output");
  for (int j = 0;j<10;j++){
    printf("%f\n",data->result[j]);
  }
    printf("%d\n",fd);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;

  has_errors |= memcmp(data->result, ref->result, SIZE);
  // Return true if it's correct.
  return !has_errors;
}
