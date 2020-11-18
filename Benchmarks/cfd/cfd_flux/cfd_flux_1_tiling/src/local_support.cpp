#include <string.h>
#include <CL/opencl.h>
#include "support.h"
#include "my_timer.h"
#include "cfd_flux.h"

int INPUT_SIZE = sizeof(struct bench_args_t);

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;

  // 0th: initialize the timer at the beginning of the program
  timespec timer = tic();
    
    cl_mem result_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args -> result), NULL, NULL);
    
    cl_mem elements_surrounding_elements_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(args -> elements_surrounding_elements), NULL, NULL);
    cl_mem normals_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(args -> normals), NULL, NULL);
    cl_mem variables_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(args -> variables), NULL, NULL);
    cl_mem fc_momentum_x_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(args -> fc_momentum_x), NULL, NULL);
    cl_mem fc_momentum_y_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(args -> fc_momentum_y), NULL, NULL);
    cl_mem fc_momentum_z_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(args -> fc_momentum_z), NULL, NULL);
    cl_mem fc_density_energy_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(args -> fc_density_energy), NULL, NULL);

  if (!result_buffer || !elements_surrounding_elements_buffer || !normals_buffer || !variables_buffer || !fc_momentum_x_buffer || !fc_momentum_y_buffer || !fc_momentum_z_buffer || !fc_density_energy_buffer)
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
    
    err |= clEnqueueWriteBuffer(commands, elements_surrounding_elements_buffer  , CL_TRUE, 0, sizeof(args -> elements_surrounding_elements), args -> elements_surrounding_elements  , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, normals_buffer  , CL_TRUE, 0, sizeof(args -> normals), args -> normals  , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, variables_buffer  , CL_TRUE, 0, sizeof(args -> variables), args -> variables  , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, fc_momentum_x_buffer  , CL_TRUE, 0, sizeof(args -> fc_momentum_x), args -> fc_momentum_x  , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, fc_momentum_y_buffer  , CL_TRUE, 0, sizeof(args -> fc_momentum_y), args -> fc_momentum_y  , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, fc_momentum_z_buffer  , CL_TRUE, 0, sizeof(args -> fc_momentum_z), args -> fc_momentum_z  , 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, fc_density_energy_buffer  , CL_TRUE, 0, sizeof(args -> fc_density_energy), args -> fc_density_energy  , 0, NULL, NULL);

  if (err != CL_SUCCESS)
  {
      printf("Error: Failed to write to device memory!\n");
      printf("Test failed\n");
      exit(1);
  }

  // 2nd: time of pageable-pinned memory copy
  toc(&timer, "memory copy");
    
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &result_buffer);

    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &elements_surrounding_elements_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &normals_buffer);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &variables_buffer);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &fc_momentum_x_buffer);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &fc_momentum_y_buffer);
    err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &fc_momentum_z_buffer);
    err |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &fc_density_energy_buffer);

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
char[]: elements_surrounding_elements
%% Section 2
char[]: normals
%% Section 3
char[]: variables
%% Section 4
char[]: fc_momentum_x
%% Section 5
char[]: fc_momentum_y
%% Section 6
char[]: fc_momentum_z
%% Section 7
char[]: fc_density_energy
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
  STAC(parse_,TYPE,_array)(s, data -> elements_surrounding_elements, SIZE * NNB);

  s = find_section_start(p,2);
  STAC(parse_,TYPE,_array)(s, data -> normals, SIZE * NNB * NDIM);

  s = find_section_start(p,3);
  STAC(parse_,TYPE,_array)(s, data -> variables, SIZE * NVAR);

  s = find_section_start(p,4);
  STAC(parse_,TYPE,_array)(s, data -> fc_momentum_x, SIZE * NDIM);

  s = find_section_start(p,5);
  STAC(parse_,TYPE,_array)(s, data -> fc_momentum_y, SIZE * NDIM);

  s = find_section_start(p,6);
  STAC(parse_,TYPE,_array)(s, data -> fc_momentum_z, SIZE * NDIM);

  s = find_section_start(p,7);
  STAC(parse_,TYPE,_array)(s, data -> fc_density_energy, SIZE * NDIM);



//printf("%d FD:++++++++++++++++++++++++++++", fd);

  for (i = 0; i < SIZE * NVAR; i++) {
      data -> result[i] = 0;
  }

}

// void data_to_input(int fd, void *vdata) {
//   struct bench_args_t *data = (struct bench_args_t *)vdata;

//   write_section_header(fd);
//   STAC(write_, TYPE, _array)(fd, data -> result, SIZE);
  
//   write_section_header(fd);
//   STAC(write_, TYPE, _array)(fd, data -> variables, SIZE * NVAR);

//   write_section_header(fd);
//   STAC(write_, TYPE, _array)(fd, data -> fc_momentum_x, SIZE);

//   write_section_header(fd);
// }


void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->result, SIZE * NVAR);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  FILE* fid = fopen("output.data", "w");

  for (int kk = 0; kk <SIZE * NVAR;kk++){
    fprintf(fid,"%.40f\n", data->result[kk]);
  }

  fclose(fid);

  printf("+++++++++++++++++++++++++++++++++++data_to_output");
  for (int j = 0;j<10;j++){
  printf("%f\n",data->result[j]);
  }
  printf("%d\n",fd);
  //  write_section_header(fd);
  //  STAC(write_,TYPE,_array)(fd, data->result, SIZE);
  //  write_float_array(fd, data->variables, 100000);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;

// for (int j = 0;j<1024*32;j++){
//     if (data->result[j]-ref->result[j]){
//     printf("%.18f\n",data->result[j]-ref->result[j]);
//     }
// }
  has_errors |= memcmp(data->result, ref->result, SIZE * NVAR);
  // Return true if it's correct.
  return !has_errors;
}
