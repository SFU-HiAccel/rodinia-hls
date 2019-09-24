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
//  int num_jobs = 1 << 16;
    //FILE* fid_re = fopen("./re_in","w");
    //FILE* fid_te = fopen("./te_in","w");
    //FILE* fid_po = fopen("./po_in","w");

    //if(!fid_re|!fid_te|!fid_po) printf("FILE error");

//    for (int k = 0; k < 10; k++) {
//    printf("%.18f\n", args -> temp[k] );
//    printf("%.18f\n", args -> power[k]);
//    }


    //fclose(fid_re);
    //fclose(fid_te);
    //fclose(fid_po);
//
//  char* seqA_batch = (char *)malloc(sizeof(args->seqA) * num_jobs);
//  char* seqB_batch = (char *)malloc(sizeof(args->seqB) * num_jobs);
//  char* alignedA_batch = (char *)malloc(sizeof(args->alignedA) * num_jobs);
//  char* alignedB_batch = (char *)malloc(sizeof(args->alignedB) * num_jobs);
//  int i;
//  for (i=0; i<num_jobs; i++) {
//    memcpy(seqA_batch + i*sizeof(args->seqA), args->seqA, sizeof(args->seqA));
//    memcpy(seqB_batch + i*sizeof(args->seqB), args->seqB, sizeof(args->seqB));
//    memcpy(alignedA_batch + i*sizeof(args->alignedA), args->alignedA, sizeof(args->alignedA));
//    memcpy(alignedB_batch + i*sizeof(args->alignedB), args->alignedB, sizeof(args->alignedB));
//  }

  // 0th: initialize the timer at the beginning of the program
  timespec timer = tic();
  // Create device buffers
  //
  //cl_mem seqA_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->seqA)*num_jobs, NULL, NULL);
  //cl_mem seqB_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->seqB)*num_jobs, NULL, NULL);
  //cl_mem alignedA_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->alignedA)*num_jobs, NULL, NULL);
  //cl_mem alignedB_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->alignedB)*num_jobs, NULL, NULL);
  
  
    cl_mem result_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args -> temp) , NULL, NULL);
    cl_mem temp_buffer   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args -> temp) , NULL, NULL);
    cl_mem power_buffer  = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(args -> power), NULL, NULL);

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
  //err = clEnqueueWriteBuffer(commands, seqA_buffer, CL_TRUE, 0, sizeof(args->seqA)*num_jobs, seqA_batch, 0, NULL, NULL);
  //err |= clEnqueueWriteBuffer(commands, seqB_buffer, CL_TRUE, 0, sizeof(args->seqB)*num_jobs, seqB_batch, 0, NULL, NULL);
  
  
    //err  = clEnqueueWriteBuffer(commands, result_buffer, CL_TRUE, 0, grid_rows * grid_cols * sizeof(float), args -> , 0, NULL, NULL);
    err  = clEnqueueWriteBuffer(commands, temp_buffer  , CL_TRUE, 0, sizeof(args -> temp) , args -> temp  , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, power_buffer , CL_TRUE, 0, sizeof(args -> power), args -> power , 0, NULL, NULL);


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
  //err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &seqA_buffer);
  //err  |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &seqB_buffer);
  //err  |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &alignedA_buffer);
  //err  |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &alignedB_buffer);
  //err  |= clSetKernelArg(kernel, 4, sizeof(int), &num_jobs);
  
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &result_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &temp_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &power_buffer);


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
  
  //err = clEnqueueReadBuffer( commands, alignedA_buffer, CL_TRUE, 0, sizeof(args->alignedA)*num_jobs, alignedA_batch, 0, NULL, NULL );  
  //err |= clEnqueueReadBuffer( commands, alignedB_buffer, CL_TRUE, 0, sizeof(args->alignedB)*num_jobs, alignedB_batch, 0, NULL, NULL );  
  

  
    //err  = clEnqueueReadBuffer(commands, ((1&SIM_TIME) ? result_buffer : temp_buffer),  CL_TRUE, 0, sizeof(args -> temp)  , args -> temp  , 0, NULL, NULL);  
    //err  = clEnqueueReadBuffer(commands, result_buffer,  CL_TRUE, 0, sizeof(args -> temp)  , args -> temp  , 0, NULL, NULL);  
    err  = clEnqueueReadBuffer(commands, temp_buffer,  CL_TRUE, 0, sizeof(args -> temp)  , args -> temp  , 0, NULL, NULL);  
    //err |= clEnqueueReadBuffer(commands, result_buffer, CL_TRUE, 0, sizeof(args -> result), result, 0, NULL, NULL);  

//for (int j = 0;j<10;j++){
//printf("%f\n",args -> temp[j]);
//}

  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    exit(1);
 }

  // 5th: time of data retrieving (PCIe + memcpy)
  toc(&timer, "data retrieving");

  //memcpy(args->alignedA, alignedA_batch, sizeof(args->alignedA));
  //memcpy(args->alignedB, alignedB_batch, sizeof(args->alignedB));
  //free(seqA_batch);
  //free(seqB_batch);
  //free(alignedA_batch);
  //free(alignedB_batch);

  //memcpy(args -> temp, (1&SIM_TIME)? temp : result, sizeof(args -> temp));

  clReleaseMemObject(cl_mem result_buffer);
  clReleaseMemObject(cl_mem temp_buffer);
  clReleaseMemObject(cl_mem power_buffer);
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
//printf("%d FD:++++++++++++++++++++++++++++", fd);
//printf("%.18f++++++++++++++++++++++",data -> temp[1]);

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
fprintf(fid,"%.8f\n", data->temp[kk]);
}

fclose(fid);






printf("+++++++++++++++++++++++++++++++++++data_to_output");
for (int j = 0;j<10;j++){
printf("%f\n",data->temp[j]);
}
printf("%d\n",fd);
//  write_section_header(fd);
//  STAC(write_,TYPE,_array)(fd, data->temp, GRID_ROWS * GRID_COLS);
//  write_float_array(fd, data->temp, 100000);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;

  has_errors |= memcmp(data->temp, ref->temp, GRID_ROWS * GRID_COLS);
  // Return true if it's correct.
  return !has_errors;
}
