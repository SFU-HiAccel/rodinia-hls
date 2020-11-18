#include <string.h>
//#include <CL/opencl.h>
//#include <CL/cl_ext_xilinx.h>
#include <CL/cl_ext.h>
#include "support.h"
#include "my_timer.h"
#include "hotspot.h"

void run_benchmark( void *vargs, cl_context&, cl_command_queue&, cl_program&, cl_kernel& );
extern void input_to_data(int fd, void *vdata);
//void data_to_input(int fd, void *vdata);
extern void output_to_data(int fd, void *vdata);
extern void data_to_output(int fd, void *vdata);
extern int check_data(void *vdata, void *vref);
int INPUT_SIZE = sizeof(struct bench_args_t);

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;

  // 0th: initialize the timer at the beginning of the program
  timespec timer = tic();

  // Create device buffers
  cl_mem_ext_ptr_t temp_ext; temp_ext.flags = XCL_MEM_DDR_BANK0;  temp_ext.param = 0; temp_ext.obj = 0;
  cl_mem temp_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX , sizeof(args -> temp) , &temp_ext, NULL);

  cl_mem_ext_ptr_t power_ext; power_ext.flags = XCL_MEM_DDR_BANK1;  power_ext.param = 0; power_ext.obj = 0;
  cl_mem power_buffer = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX , sizeof(args -> power) , &power_ext, NULL);
   
  cl_mem_ext_ptr_t result_ext; result_ext.flags = XCL_MEM_DDR_BANK2; result_ext.param = 0; result_ext.obj = 0;
  cl_mem result_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX , sizeof(args -> temp) , &result_ext, NULL);

  if (!result_buffer || !temp_buffer || !power_buffer)
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
  
  // Set the arguments to our compute kernel
  // CHange the sequence of setting arguments and memory copy
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &result_buffer);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &temp_buffer);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &power_buffer);

  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to set kernel arguments! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }  

  // 2nd: time of setting arguments
  toc(&timer, "set arguments");

  err  = clEnqueueWriteBuffer(commands, temp_buffer  , CL_TRUE, 0, sizeof(args -> temp) , args -> temp  , 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, power_buffer , CL_TRUE, 0, sizeof(args -> power), args -> power , 0, NULL, NULL);

  if (err != CL_SUCCESS)
  {
      printf("Error: Failed to write to device memory!\n");
      printf("Test failed\n");
      exit(1);
  }

  // 3rd: time of pageable-pinned memory copy
  toc(&timer, "memory copy");
  
  // Execute the kernel over the entire range of our 1d input data set
  // using the maximum number of work group items for this device

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
  err  = clEnqueueReadBuffer(commands, temp_buffer,  CL_TRUE, 0, sizeof(args -> temp)  , args -> temp  , 0, NULL, NULL);  

  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    exit(1);
 }

  // 5th: time of data retrieving (PCIe + memcpy)
  toc(&timer, "data retrieving");

  clReleaseMemObject(temp_buffer);
  clReleaseMemObject(result_buffer);
  clReleaseMemObject(power_buffer);
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
  STAC(parse_,TYPE,_array)(s, data -> temp, GRID_ROWS*GRID_COLS);

  s = find_section_start(p,2);
  STAC(parse_,TYPE,_array)(s, data -> power, GRID_ROWS*GRID_COLS);
}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_, TYPE, _array)(fd, data -> temp, GRID_ROWS * GRID_COLS);

  write_section_header(fd);
  STAC(write_, TYPE, _array)(fd, data -> power, GRID_ROWS * GRID_COLS);

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
  STAC(parse_,TYPE,_array)(s, data->temp, GRID_ROWS * GRID_COLS);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  FILE* fid = fopen("output.data", "w");

  for (int kk = 0; kk <GRID_ROWS*GRID_COLS;kk++){
    fprintf(fid,"%.18f\n", data->temp[kk]);
  }

  fclose(fid);

  printf("+++++++++++++++++++++++++++++++++++data_to_output");
  for (int j = 0;j<10;j++){
    printf("%f\n",data->temp[j]);
  }

  printf("%d\n",fd);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;

  // has_errors |= memcmp(data->temp, ref->temp, GRID_ROWS * GRID_COLS);

  for(int i = 0; i < GRID_COLS * GRID_ROWS; i++){
    if(data->temp[i] != ref->temp[i]){
      float tmp;
      if(data->temp[i] > ref -> temp[i]){
        tmp = data->temp[i] - ref->temp[i];
      } else {
        tmp = ref->temp[i] - data->temp[i];
      }

      float error_rate = tmp / ref->temp[i];

      if(error_rate > 0.005)
        has_errors++;
    }
  }
  // Return true if it's correct.
  return !has_errors;
}
