#include "lc_dilate.h"
#include "support.h"
#include <string.h>
#include "my_timer.h"
#include <CL/opencl.h>

int INPUT_SIZE = sizeof(struct bench_args_t);

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;

  // 0th: initialize the timer at the beginning of the program
  timespec timer = tic();
    
    cl_mem result_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args -> result), NULL, NULL);
    cl_mem img_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(args -> img), NULL, NULL);

  if (!result_buffer || !img_buffer)
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
    err |= clEnqueueWriteBuffer(commands, img_buffer  , CL_TRUE, 0, sizeof(args -> img), args -> img  , 0, NULL, NULL);

  if (err != CL_SUCCESS)
  {
      printf("Error: Failed to write to device memory!\n");
      printf("Test failed\n");
      exit(1);
  }

  // 2nd: time of pageable-pinned memory copy
  toc(&timer, "memory copy");
    
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &result_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &img_buffer);

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
char[]: img
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
  STAC(parse_,TYPE,_array)(s, data -> img, GRID_ROWS*GRID_COLS);

for (i = 0; i < GRID_COLS * GRID_ROWS; i++) {
    data -> result[i] = 0;
}

}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_, TYPE, _array)(fd, data -> result, GRID_ROWS * GRID_COLS);
  
  write_section_header(fd);
  STAC(write_, TYPE, _array)(fd, data -> img, GRID_ROWS * GRID_COLS);


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
  STAC(parse_,TYPE,_array)(s, data->result, GRID_ROWS * GRID_COLS);
}




void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;


FILE* fid = fopen("output.data", "w");

for (int kk = 0; kk <GRID_ROWS*GRID_COLS;kk++){
fprintf(fid,"%.18f\n", data->result[kk]);
}

fclose(fid);



printf("+++++++++++++++++++++++++++++++++++data_to_output");
for (int j = 0;j<10;j++){
printf("%f\n",data->result[j]);
}
//printf("%d\n",fd);
//  write_section_header(fd);
//  STAC(write_,TYPE,_array)(fd, data->result, GRID_ROWS * GRID_COLS);
//  write_float_array(fd, data->img, 100000);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;

  has_errors |= memcmp(data->result, ref->result, GRID_ROWS * GRID_COLS);
  // Return true if it's correct.
  return !has_errors;
}
