#include <string.h>
#include <CL/opencl.h>
#include "support.h"
#include "my_timer.h"
#include "knn.h"

#include <string.h>
#include <CL/opencl.h>
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
  cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args -> input_query) , NULL, NULL);
  cl_mem search_buffer   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args -> search_space_data) , NULL, NULL);
  cl_mem distance_buffer  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args -> distance), NULL, NULL);

  if (!input_buffer || !search_buffer || !distance_buffer)
  {
    printf("Error: Failed to allocate device memory!\n");
    printf("Test failed\n");
    exit(1);
  }    

  // 1st: time of buffer allocation
  toc(&timer, "buffer allocation");

  // Write our data set into device buffers
  int err;

  err  = clEnqueueWriteBuffer(commands, input_buffer  , CL_TRUE, 0, sizeof(args -> input_query) , args -> input_query  , 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, search_buffer , CL_TRUE, 0, sizeof(args -> search_space_data), args -> search_space_data , 0, NULL, NULL);


  if (err != CL_SUCCESS){
      printf("Error: Failed to write to device memory!\n");
      printf("Test failed\n");
      exit(1);
  }

  // 2nd: time of pageable-pinned memory copy
  toc(&timer, "memory copy");
    
  // Set the arguments to our compute kernel  
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &search_buffer);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &distance_buffer);


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
  err  = clEnqueueReadBuffer(commands, distance_buffer,  CL_TRUE, 0, sizeof(args -> distance)  , args -> distance  , 0, NULL, NULL);   

  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }

  // 5th: time of data retrieving (PCIe + memcpy)
  toc(&timer, "data retrieving");

  clReleaseMemObject(input_buffer);
  clReleaseMemObject(search_buffer);
  clReleaseMemObject(distance_buffer);

  //free(result_buffer);
}

/* Input format:
%% Section 1
char[]: sequence A
%% Section 2
char[]: sequence B
*/

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  for(int i = 0; i < NUM_FEATURE; ++i){
  	data->input_query[i] = i + 1.0;
  }

  for(int i = 0; i<NUM_PT_IN_SEARCHSPACE*NUM_FEATURE; ++i){
  	data->search_space_data[i] = i + 1.0;
  }

  // char *p, *s;
  // // Zero-out everything.
  // memset(vdata,0,sizeof(struct bench_args_t));
  // // Load input string
  // p = readfile(fd);
  // s = find_section_start(p,1);
  // STAC(parse_,TYPE,_array)(s, data -> temp, GRID_ROWS*GRID_COLS);

  // s = find_section_start(p,2);
  // STAC(parse_,TYPE,_array)(s, data -> power, GRID_ROWS*GRID_COLS);

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

  for(int i = 0; i < NUM_FEATURE; ++i){
  	data->input_query[i] = i + 1.0;
  }

  for(int i = 0; i<NUM_PT_IN_SEARCHSPACE*NUM_FEATURE; ++i){
  	data->search_space_data[i] = i + 1.0;
  }

  float sum = 0.0;
  float delta = 0.0;
  for(int i = 0; i < NUM_PT_IN_SEARCHSPACE; i++ ) {
  	sum = 0.0;
  	for (int j = 0; j < NUM_FEATURE; ++j){
  		delta = data->search_space_data[i*NUM_FEATURE+j] - data->input_query[j];
	    sum += delta * delta;
		}

		data->distance[i] = sum;
	}
  // char *p, *s;

  // // Zero-out everything.
  // memset(vdata,0,sizeof(struct bench_args_t));

  // // Load input string
  // p = readfile(fd);

  // s = find_section_start(p,1);
  // STAC(parse_,TYPE,_array)(s, data->temp, GRID_ROWS * GRID_COLS);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;


  FILE* fid = fopen("output.data", "w");

  for (int kk = 0; kk <GRID_ROWS*GRID_COLS;kk++){
    fprintf(fid,"%.7f\n", data->temp[kk]);
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

  for(int i = 0; i < NUM_PT_IN_SEARCHSPACE; i++){
  	if(data->distance[i] != ref->distance[i]){
  		has_errors++;
  	}
  }

  //has_errors |= memcmp(data->temp, ref->temp, GRID_ROWS * GRID_COLS);

  // Return true if it's correct.
  return !has_errors;
}
