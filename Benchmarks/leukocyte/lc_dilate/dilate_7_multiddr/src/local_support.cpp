#include "dilate.h"
#include "support.h"
#include "my_timer.h"
#include <string.h>

int INPUT_SIZE = sizeof(struct bench_args_t);
 

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdio.h>									// (in directory known to compiler)			needed by printf, stderr
#include <limits.h>									// (in directory known to compiler)			needed by INT_MIN, INT_MAX
// #include <sys/time.h>							// (in directory known to compiler)			needed by ???
#include <math.h>									// (in directory known to compiler)			needed by log, pow
#include <string.h>									// (in directory known to compiler)			needed by memset

//======================================================================================================================================================150
//	END
//======================================================================================================================================================150

void run_benchmark(void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel){
	struct bench_args_t *args = (struct bench_args_t *)vargs;

	timespec timer = tic();

	cl_mem_ext_ptr_t result_ext; result_ext.flags = XCL_MEM_DDR_BANK0; result_ext.param=0; result_ext.obj=0;
	cl_mem result_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, sizeof(args -> result), &result_ext, NULL);

	cl_mem_ext_ptr_t img_ext; img_ext.flags = XCL_MEM_DDR_BANK1; img_ext.param=0; img_ext.obj=0;
	cl_mem img_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, sizeof(args -> img), &img_ext, NULL);

	if(!result_buffer || !img_buffer){
		printf("Error: Failed to write to device memory!\n");
		printf("Test failed\n");
    	exit(1);
	}

	//Buffer allocation time
	toc(&timer, "buffer allocation");

	int err;
	err = clEnqueueWriteBuffer(commands, img_buffer, CL_TRUE, 0, sizeof(args -> img), args -> img, 0, NULL, NULL);

	if (err != CL_SUCCESS){
      printf("Error: Failed to write to device memory!\n");
      printf("Test failed\n");
      exit(1);
  	}	

  	//Buffer copyt time
  	toc(&timer, "memory copy");

  	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &result_buffer);
  	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &img_buffer);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);
		printf("Test failed\n");
		exit(1);
	}

	//Setting arguments time
	toc(&timer, "set arguments");

#ifdef C_KERNEL
  	err = clEnqueueTask(commands, kernel, 0, NULL, NULL);
#else
	printf("Error: OpenCL kernel is not currently supported!\n");
	exit(1);
#endif
	if (err){
		printf("Error: Failed to execute kernel! %d\n", err);
		printf("Test failed\n");
		exit(1);
	}

	clFinish(commands);
	toc(&timer, "kernel execution");

	err  = clEnqueueReadBuffer(commands, result_buffer, CL_TRUE, 0, sizeof(args -> result), args -> result, 0, NULL, NULL); 

	if (err != CL_SUCCESS){
		printf("Error: Failed to read output array! %d\n", err);
		printf("Test failed\n");
		exit(1);
	}

  	// 5th: time of data retrieving (PCIe + memcpy)
  	toc(&timer, "data retrieving");

}

void input_to_data(int fd, void *vdata) {
	struct bench_args_t *data = (struct bench_args_t *)vdata;
	char *p, *s;
	int i;
	// Zero-out everything.
	memset(vdata,0,sizeof(struct bench_args_t));
	// Load input string
	p = readfile(fd);
	s = find_section_start(p,1);

	float tmp_img[GRID_ROWS * GRID_COLS];
	STAC(parse_,TYPE,_array)(s, tmp_img, GRID_ROWS*GRID_COLS);

	int starting_idx = MAX_RADIUS*GRID_COLS;
	for (int i(0); i<GRID_ROWS; ++i){
		for (int j(0); j<GRID_COLS; ++j){
			data -> img[starting_idx+(i*GRID_COLS+j)] = tmp_img[i*GRID_COLS+j];
		}
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
	printf("start reading check data\n");

	p = readfile(fd);

	printf("finish reading check data\n");
	
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

	printf("%d\n",fd);
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

	printf("%d \n", has_errors);
	return !has_errors;
}

