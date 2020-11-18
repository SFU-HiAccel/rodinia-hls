#include <string.h>
#include <CL/opencl.h>
#include "support.h"
#include "my_timer.h"
#include "kmeans.h"

int INPUT_SIZE = sizeof(struct bench_args_t);

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel){
	struct bench_args_t *args = (struct bench_args_t *)vargs;

	timespec timer = tic();

	cl_mem_ext_ptr_t feature_ext; feature_ext.flags = XCL_MEM_DDR_BANK0; feature_ext.param=0; feature_ext.obj=0;
	cl_mem feature_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, sizeof(args->FEATURE), &feature_ext, NULL);

	cl_mem_ext_ptr_t cluster_ext; cluster_ext.flags = XCL_MEM_DDR_BANK1; cluster_ext.param=0; cluster_ext.obj=0;
	cl_mem cluster_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, sizeof(args->CLUSTER), &cluster_ext, NULL);

	cl_mem_ext_ptr_t membership_ext; membership_ext.flags = XCL_MEM_DDR_BANK2; membership_ext.param=0; membership_ext.obj=0;
	cl_mem membership_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, sizeof(args->MEMBERSHIP), &membership_ext, NULL);

	if (!feature_buffer || !cluster_buffer || !membership_buffer)
	{
		printf("Error: Failed to allocate device memory!\n");
		printf("Test failed\n");
		exit(1);
	}    

	toc(&timer, "buffer allocation");

	int err;

	err = clEnqueueWriteBuffer(commands, feature_buffer, CL_TRUE, 0, sizeof(args -> FEATURE), args -> FEATURE, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, cluster_buffer, CL_TRUE, 0, sizeof(args -> CLUSTER), args -> CLUSTER, 0, NULL, NULL);


	if (err != CL_SUCCESS){
		printf("Error: Failed to write to device memory!\n");
		printf("Test failed\n");
		exit(1);
	}

	toc(&timer, "memory copy");

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &feature_buffer);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cluster_buffer);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &membership_buffer);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);
		printf("Test failed\n");
		exit(1);
	}

	toc(&timer, "set arguments");

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

	clFinish(commands);
	toc(&timer, "kernel execution");

	err = clEnqueueReadBuffer(commands, membership_buffer, CL_TRUE, 0, sizeof(args -> MEMBERSHIP), args -> MEMBERSHIP, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		printf("Test failed\n");
		exit(1);
	}

	toc(&timer, "data retrieving");

	clReleaseMemObject(feature_buffer);
	clReleaseMemObject(cluster_buffer);
	clReleaseMemObject(membership_buffer);
}

void input_to_data(int fd, void *vdata) {
	struct bench_args_t *data = (struct bench_args_t *)vdata;

	float low = -10.0;
	float high = 10.0;

	for (int i(0); i<NPOINTS*NFEATURES; ++i){
		data -> FEATURE[i] = low + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(high-low)));
	}

	for (int i(0); i<NCLUSTERS*NFEATURES; ++i){
		data -> CLUSTER[i] = low + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(high-low)));
	}
}

void data_to_input(int fd, void *vdata) {

}

void data_to_output(int fd, void *vdata) {

}

void output_to_data(int fd, void *vdata) {

}

int check_data( void *vdata, void *vref ) {
	int has_errors = 0;

	struct bench_args_t *data = (struct bench_args_t *)vdata;

	int SW_MEMBERSHIP[NPOINTS];

	for (int i(0); i<NPOINTS; i++)
    {
        float min_dist = FLT_MAX;

        int index = 0;
        /* find the cluster center id with min distance to pt */
        for (int j(0); j<NCLUSTERS; j++) {
            float dist = 0.0;

            for (int k(0); k<NFEATURES; k++) {
                float diff = data -> FEATURE[NFEATURES*i+k] - data -> CLUSTER[NFEATURES*j+k];
                dist += diff*diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                index = j;
            }
        }

        /* assign the membership to object i */
        SW_MEMBERSHIP[i] = index;
    }

    for (int i(0); i<NPOINTS; ++i){
		if (SW_MEMBERSHIP[i] != data -> MEMBERSHIP[i])
			has_errors++;
	}

	return !has_errors;
}
