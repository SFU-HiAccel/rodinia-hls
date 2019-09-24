#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "support.h"
#include "my_timer.h"
#include "lavaMD.h"

int INPUT_SIZE = sizeof(struct bench_args_t);

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) 
{
  struct bench_args_t *args = (struct bench_args_t *)vargs;

#ifdef DEBUG
  //print-out
  int row, col;
  for (row = 0; row < N; ++row)
  {
    for (col = 0; col < POS_DIM; ++col)
    {
      printf("%f, ", args->pos_i[row*POS_DIM+col]);
    }
    printf("\n");
  }
#endif

  const int POS_I_SIZE = sizeof(TYPE) * DIMENSION_3D_PADDED * NUMBER_PAR_PER_BOX * POS_DIM;
  const int Q_I_SIZE = sizeof(TYPE) * DIMENSION_3D_PADDED * NUMBER_PAR_PER_BOX;
  const int POS_O_SIZE = sizeof(TYPE) * DIMENSION_3D * NUMBER_PAR_PER_BOX * POS_DIM;
  const int LINE_1D_SIZE = sizeof(TYPE) * DIMENSION_1D * NUMBER_PAR_PER_BOX;
  const int LINE_1D_LEN = DIMENSION_1D * NUMBER_PAR_PER_BOX;

  // Format raw input data 
  //
  TYPE* pos_i_padded = (TYPE *)malloc(POS_I_SIZE);
  TYPE* q_i_padded = (TYPE *)malloc(Q_I_SIZE);
  TYPE* pos_o_padded = (TYPE *)malloc(POS_O_SIZE);
  // Zero-out everything
  memset(pos_i_padded, 0, POS_I_SIZE);
  memset(q_i_padded, 0, Q_I_SIZE);
  memset(pos_o_padded, 0, POS_O_SIZE);

#ifdef DEBUG
  // Debuggin Print-outs
  printf("sizeof(args->pos_i): %d \n", sizeof(args->pos_i));
  printf("sizeof(args->q_i): %d \n", sizeof(args->q_i));
  printf("sizeof(TYPE): %d \n", sizeof(TYPE));
  printf("sizeof(args->pos_i): %d \n", sizeof(TYPE) * DIMENSION_3D_PADDED * POS_DIM * NUMBER_PAR_PER_BOX);
  printf("sizeof(args->q_i): %d \n", sizeof(TYPE) * DIMENSION_3D_PADDED * NUMBER_PAR_PER_BOX);
  printf("sizeof(pos_i_padded): %d \n", sizeof(pos_i_padded));
  printf("sizeof(q_i_padded): %d \n", sizeof(q_i_padded));
  printf("\n");

  // memcpy( pos_i_padded, args->pos_i, sizeof(args->pos_i) ); //pos_i
  // memcpy( q_i_padded, args->q_i, sizeof(args->q_i) ); //q_i
  // for (i=0; i<10; ++i)
  //   printf("args->pos_i[%d]: %f \n", i, args->pos_i[i]);
  // memcpy( pos_i_padded + 1, args->pos_i + 1, LINE_1D_SIZE*POS_DIM ); //pos_i
  // for (i=0; i<10; ++i)
  //   printf("pos_i_padded[%d]: %f \n", i, pos_i_padded[i]);
#endif

  // Formatting - skipping over padded cells
  //
  int i, j, k;
  int base_addr_q = 0;
  int base_addr_pos = 0;
  int num_lines = 0;

#ifdef DEBUG
  //Verification
  for ( i=0; i<(DIMENSION_3D_PADDED)*NUMBER_PAR_PER_BOX; ++i )
  {
    if (q_i_padded[i] != 0.0)
    {
      printf("ERROR!!! - staring point\n");
      return;
    }
  }
#endif

  base_addr_q = DIMENSION_2D_PADDED * NUMBER_PAR_PER_BOX;
  base_addr_pos = DIMENSION_2D_PADDED * NUMBER_PAR_PER_BOX * POS_DIM;
  for (i=0; i<DIMENSION_1D; ++i)
  {
    base_addr_q += (DIMENSION_1D_PADDED+1) * NUMBER_PAR_PER_BOX;
    base_addr_pos += (DIMENSION_1D_PADDED+1) * NUMBER_PAR_PER_BOX * POS_DIM;
    for (j=0; j<DIMENSION_1D; ++j)
    {
      num_lines = i*DIMENSION_1D+j;
      memcpy( q_i_padded + base_addr_q, args->q_i + num_lines*LINE_1D_LEN, LINE_1D_SIZE ); //q_i
      memcpy( pos_i_padded + base_addr_pos, args->pos_i + num_lines*LINE_1D_LEN*POS_DIM, LINE_1D_SIZE*POS_DIM ); //pos_i
      base_addr_q += (DIMENSION_1D+2) * NUMBER_PAR_PER_BOX;
      base_addr_pos += (DIMENSION_1D+2) * NUMBER_PAR_PER_BOX*POS_DIM;
    }
    base_addr_q += (DIMENSION_1D_PADDED-1) * NUMBER_PAR_PER_BOX;
    base_addr_pos += (DIMENSION_1D_PADDED-1) * NUMBER_PAR_PER_BOX * POS_DIM;
  }


#ifdef DEBUG
// base_addr_q = DIMENSION_2D_PADDED * NUMBER_PAR_PER_BOX;
// base_addr_q += (DIMENSION_1D_PADDED+1) * NUMBER_PAR_PER_BOX;
//   memcpy( q_i_padded + base_addr_q, args->q_i, sizeof(float)*10*100 ); //q_i
// base_addr_q += 12 * NUMBER_PAR_PER_BOX;
//   memcpy( &(q_i_padded[base_addr_q]), args->q_i, sizeof(float)*10*100 ); //q_i

  //Verification
  for ( i=0; i<(DIMENSION_2D_PADDED+DIMENSION_1D_PADDED+1)*NUMBER_PAR_PER_BOX; ++i )
  {
    if (q_i_padded[i] != 0.0)
    {
      printf("ERROR!!! - bottom\n");
      return;
    }
  }
  int base_addr = (DIMENSION_2D_PADDED)*NUMBER_PAR_PER_BOX;
  for (i=0; i<DIMENSION_1D; ++i)
  { 
    base_addr += (DIMENSION_1D_PADDED+1)*NUMBER_PAR_PER_BOX;
    for (j=0; j<DIMENSION_1D; ++j)
    {
      base_addr += DIMENSION_1D*NUMBER_PAR_PER_BOX;
      for (k=0; k<2*NUMBER_PAR_PER_BOX; ++k)
      {
        if (q_i_padded[base_addr+k] != 0.0)
        {
          printf("ERROR!!! - middle in index: %d, %d, %d\n", i, j, k);
          return;
        }
      }
      base_addr += 2*NUMBER_PAR_PER_BOX;
    }
    base_addr += (DIMENSION_1D_PADDED-1)*NUMBER_PAR_PER_BOX;
  }
  printf("base_addr: %d\n", base_addr);
  for ( i=0; i<(DIMENSION_2D_PADDED)*NUMBER_PAR_PER_BOX; ++i)
  {
    if (q_i_padded[base_addr+i] != 0.0)
    {
      printf("ERROR!!! - top @ index: %d\n", i);
      return;
    }
  }
#endif

  // 0th: initialize the timer at the beginning of the program
  timespec timer = tic();

  // Create device buffers
  //
  cl_mem pos_i_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, POS_I_SIZE, NULL, NULL);
  cl_mem q_i_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, Q_I_SIZE, NULL, NULL);
  cl_mem pos_o_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, POS_O_SIZE, NULL, NULL);
  if (!pos_i_buffer || !q_i_buffer || !pos_o_buffer)
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
  err = clEnqueueWriteBuffer(commands, pos_i_buffer, CL_TRUE, 0, POS_I_SIZE, pos_i_padded, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, q_i_buffer, CL_TRUE, 0, Q_I_SIZE, q_i_padded, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, pos_o_buffer, CL_TRUE, 0, POS_O_SIZE, pos_o_padded, 0, NULL, NULL);
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
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &pos_i_buffer);
  err  |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &q_i_buffer);
  err  |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &pos_o_buffer);
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
  err = clEnqueueReadBuffer( commands, pos_o_buffer, CL_TRUE, 0, POS_O_SIZE, args->pos_o, 0, NULL, NULL );
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }

  // 5th: time of data retrieving (PCIe + memcpy)
  toc(&timer, "data retrieving");

  free(pos_i_padded);
  free(q_i_padded);
  free(pos_o_padded);
  clReleaseMemObject(pos_i_buffer);
  clReleaseMemObject(q_i_buffer);
  clReleaseMemObject(pos_o_buffer);
}

/* Input format:
%% Section 1 - 400000 lines
%f: input distance vectors.v  
%f: input distance vectors.x
%f: input distance vectors.y
%f: input distance vectors.z
.
.
.
%f: input charges
*/
void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  s = parse_float_array2D(s, data->pos_i, N, POS_DIM);

#ifdef DEBUG
  int i=0;
  for ( i=0; i<10; ++i)
    printf("Reading input.data->pos_i[%d]: %f \n", i, data->pos_i[i]);
#endif

  STAC(parse_,TYPE,_array)(s, data->q_i, N);

#ifdef DEBUG
  int j=0;
  for ( j=0; j<10; ++j)
    printf("Reading input.data->q_i[%d]: %f \n", j, data->q_i[j]);
#endif
}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  write_float_array2D(fd, data->pos_i, N, POS_DIM); 

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->q_i, N);

  write_section_header(fd);
}

/* Output format:
%% Section 1
%f: output distance vectors.v  
%f: output distance vectors.x
%f: output distance vectors.y
%f: output distance vectors.z
*/

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  parse_float_array2D(s, data->pos_o, N, POS_DIM);

#ifdef DEBUG
  int i=0;
  for ( i=0; i<10; ++i)
    printf("Reading check.data->pos_o[%d]: %f \n", i, data->pos_o[i]);
#endif
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

#ifdef DEBUG
  int i=0;
  for ( i=0; i<10; ++i)
    printf("writing output.data->pos_o[%d]: %f \n", i, data->pos_o[i]);
#endif

  write_section_header(fd);
  write_float_array2D(fd, data->pos_o, N, POS_DIM); 
  write_section_header(fd);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;

  // has_errors |= memcmp(data->pos_o, ref->pos_o, sizeof(TYPE)*POS_DATA_SIZE_FLATTEN);

  int i=0;
  for (i=0; i<POS_DATA_SIZE_FLATTEN; ++i)
  {
    if (abs(data->pos_o[i] - ref->pos_o[i]) > 0.01)
    {
      printf("Error found @ index: %d/%d - [%f, %f]\n", i, POS_DATA_SIZE_FLATTEN, data->pos_o[i], ref->pos_o[i]);
      return 0;
    }
  }

  // Return true if it's correct.
  return !has_errors;
}
