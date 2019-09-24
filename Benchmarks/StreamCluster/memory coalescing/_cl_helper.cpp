#include "_cl_helper.h"
#include <CL/opencl.h>



//#pragma OPENCL EXTENSION cl_nv_compiler_options:enable
/*------------------------------------------------------------
	@struct:	the structure of device properties
	@date:		24/03/2011
------------------------------------------------------------*/

static cl_platform_id          platform_id;
static cl_context              context;
static cl_device_id            device;
static cl_command_queue        queue;
static cl_program              program;
static cl_int		           cl_status;
static cl_kernel               kernel;

static cl_mem                  cl_coord;
static cl_mem                  cl_weight;
static cl_mem                  cl_cost;
static cl_mem                  cl_target;
static cl_mem                  cl_assign;
static cl_mem                  cl_center_table;
static cl_mem                  cl_switch_membership;
static cl_mem                  cl_cost_of_opening_x;



/*------------------------------------------------------------
	@function:	Initlize CL objects
	@params:	
		device_id: device id
		device_type: the types of devices, e.g. CPU, GPU, ACCERLERATOR,...	
		(1) -t cpu/gpu/acc -d 0/1/2/...
		(2) -t cpu/gpu/acc [-d 0]
		(3) [-t default] -d 0/1/2/...
		(4) NULL [-d 0]
	@return:
	@description:
		there are 5 steps to initialize all the OpenCL objects needed,
	@revised: 
		get the number of devices and devices have no relationship with context
	@date:		24/03/2011
------------------------------------------------------------*/
int load_file_to_memory(const char *filename, char **result){ 
  size_t size = 0;
  FILE *f = fopen(filename, "rb");
  if (f == NULL) 
  { 
    *result = NULL;
    return -1; // -1 means file opening fail 
  } 
  fseek(f, 0, SEEK_END);
  size = ftell(f);
  fseek(f, 0, SEEK_SET);
  *result = (char *)malloc(size+1);
  if (size != fread(*result, sizeof(char), size, f)) 
  { 
    free(*result);
    return -2; // -2 means file reading fail 
  } 
  fclose(f);
  (*result)[size] = 0;
  return size;
}
void _clInit(char* bitstream){

    char cl_platform_vendor[1001];
    char cl_platform_name[1001];

    cl_int err;

    err = clGetPlatformIDs(1,&platform_id,NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to find an OpenCL platform!\n");
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }
    err = clGetPlatformInfo(platform_id,CL_PLATFORM_VENDOR,1000,(void *)cl_platform_vendor,NULL);
    if (err != CL_SUCCESS){
        printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }
    printf("CL_PLATFORM_VENDOR %s\n",cl_platform_vendor);
    err = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME,1000,(void *)cl_platform_name,NULL);
    if (err != CL_SUCCESS){
        printf("Error: clGetPlatformInfo(CL_PLATFORM_NAME) failed!\n");
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }
    printf("CL_PLATFORM_NAME %s\n",cl_platform_name);
 
    // Connect to a compute device
    int fpga = 1;
    err = clGetDeviceIDs(platform_id, fpga ? CL_DEVICE_TYPE_ACCELERATOR : CL_DEVICE_TYPE_CPU,
        1, &device, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to create a device group!\n");
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }
  
    // Create a compute context 
    //
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    if (!context || err != CL_SUCCESS){
        printf("Error: Failed to create a compute context!\n");
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }

    // Create a command queue
    //
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (!queue || err != CL_SUCCESS){
        printf("Error: Failed to create a command queue!\n");
        printf("Error: code %i\n",err);
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }

    // Create Program Objects
  
    // Load binary from disk
    unsigned char *kernelbinary;
    // The sixth parameter specifies the name of the kernel binary
    printf("loading %s\n", bitstream);
    int n_i = load_file_to_memory(bitstream, (char **) &kernelbinary);
    if (n_i < 0) {
        printf("failed to load kernel from xclbin: %s\n", bitstream);
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }
    size_t n = n_i;
    // Create the compute program from offline
    program = clCreateProgramWithBinary(context, 1, &device, &n,
        (const unsigned char **) &kernelbinary, &cl_status, &err);
    if ((!program) || (err!=CL_SUCCESS)) {
        printf("Error: Failed to create compute program from binary %d!\n", err);
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS){
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "streamcluster", &err);
    if (!kernel || err != CL_SUCCESS){
        printf("Error: Failed to create compute kernel!\n");
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }

    cl_coord = clCreateBuffer(context, CL_MEM_READ_ONLY, DIM * BATCH_SIZE * sizeof(float), NULL, &err);
    cl_weight = clCreateBuffer(context, CL_MEM_READ_ONLY, BATCH_SIZE * sizeof(float), NULL, &err);
    cl_cost = clCreateBuffer(context, CL_MEM_READ_ONLY, BATCH_SIZE * sizeof(float), NULL, &err);
    cl_assign = clCreateBuffer(context, CL_MEM_READ_ONLY, BATCH_SIZE * sizeof(int), NULL, &err);
    cl_target = clCreateBuffer(context, CL_MEM_READ_ONLY, DIM * sizeof(float), NULL, &err);
    cl_center_table = clCreateBuffer(context, CL_MEM_READ_ONLY, BATCH_SIZE * sizeof(int), NULL, &err);
    cl_switch_membership = clCreateBuffer(context, CL_MEM_WRITE_ONLY, BATCH_SIZE * sizeof(char), NULL, &err);
    cl_cost_of_opening_x = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create buffer %d!\n", err);
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }

}


double _clTask(float* coord, float* weight, float* target, float* cost, int* assign, int* center_table, char* switch_membership, float* work_mem, int num, int numcenter)
{
    cl_int err;

    cl_mem cl_work_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, numcenter * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create work mem buffer %d!\n", err);
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }

    err = clEnqueueWriteBuffer(queue, cl_coord, CL_TRUE, 0, DIM * BATCH_SIZE * sizeof(float), coord, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, cl_weight, CL_TRUE, 0, BATCH_SIZE * sizeof(float), weight, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, cl_target, CL_TRUE, 0, DIM * sizeof(float), target, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, cl_cost, CL_TRUE, 0, BATCH_SIZE * sizeof(float), cost, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, cl_assign, CL_TRUE, 0, BATCH_SIZE * sizeof(int), assign, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, cl_center_table, CL_TRUE, 0, BATCH_SIZE * sizeof(int), center_table, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, cl_work_mem, CL_TRUE, 0, numcenter * sizeof(float), work_mem, 0, NULL, NULL);

    if (err != CL_SUCCESS) {
        printf("Error: Failed to write to buffer %d!\n", err);
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }

    float cost_of_opening_x;

    int arg_count = 0;
    err = clSetKernelArg(kernel, arg_count++, sizeof(cl_mem), &cl_coord);
    err |= clSetKernelArg(kernel, arg_count++, sizeof(cl_mem), &cl_weight);
    err |= clSetKernelArg(kernel, arg_count++, sizeof(cl_mem), &cl_cost);
    err |= clSetKernelArg(kernel, arg_count++, sizeof(cl_mem), &cl_target);
    err |= clSetKernelArg(kernel, arg_count++, sizeof(cl_mem), &cl_assign);
    err |= clSetKernelArg(kernel, arg_count++, sizeof(cl_mem), &cl_center_table);
    err |= clSetKernelArg(kernel, arg_count++, sizeof(cl_mem), &cl_switch_membership);
    err |= clSetKernelArg(kernel, arg_count++, sizeof(cl_mem), &cl_work_mem);
    err |= clSetKernelArg(kernel, arg_count++, sizeof(int), &num);
    err |= clSetKernelArg(kernel, arg_count++, sizeof(cl_mem), &cl_cost_of_opening_x);
    err |= clSetKernelArg(kernel, arg_count++, sizeof(int), &numcenter);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set arguments %d!\n", err);
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }

    err = clEnqueueTask(queue, kernel, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to execute kernel %d!\n", err);
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }
    clFinish(queue);

    err = clEnqueueReadBuffer(queue, cl_work_mem, CL_TRUE, 0, numcenter * sizeof(float), work_mem, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(queue, cl_switch_membership, CL_TRUE, 0, BATCH_SIZE * sizeof(char), switch_membership, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(queue, cl_cost_of_opening_x, CL_TRUE, 0, sizeof(float), &cost_of_opening_x, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read buffer %d!\n", err);
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }

    err = clReleaseMemObject(cl_work_mem);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to release work mem buffer %d!\n", err);
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }

    return (double)cost_of_opening_x;

}

void _clClean()
{
    cl_int err;

    err = clReleaseMemObject(cl_coord);
    err |= clReleaseMemObject(cl_weight);
    err |= clReleaseMemObject(cl_cost);
    err |= clReleaseMemObject(cl_target);
    err |= clReleaseMemObject(cl_assign);
    err |= clReleaseMemObject(cl_switch_membership);
    err |= clReleaseMemObject(cl_center_table);
    err |= clReleaseMemObject(cl_cost_of_opening_x);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to release buffer %d!\n", err);
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }

    err = clReleaseProgram(program);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to release program %d!\n", err);
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }
    err = clReleaseKernel(kernel);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to release kernel %d!\n", err);
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }
    err = clReleaseCommandQueue(queue);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to release queue %d!\n", err);
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }
    err = clReleaseContext(context);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to release context %d!\n", err);
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }
}