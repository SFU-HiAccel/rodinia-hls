#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include "kmeans.h"
#include "my_timer.h"
#include <CL/opencl.h>
#define COALESCE_SIZE 48

extern int setup(struct bench_args_t *args, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel);

int INPUT_SIZE = sizeof(struct bench_args_t);

void kmeansFPGA(float **feature,    /* in: [npoints][nfeatures] */
              int     nfeatures,  /* in */
              int     npoints,/* in */
              int     nclusters,/* in */
              float   threshold,
              float **clusters,   /* out */
              int *membership,
                cl_context& context,
                cl_command_queue& commands,
                cl_program& program,
                cl_kernel& kernel)
{
  int i, j;
  float delta;
  int *prev_membership;
  float  **new_centers; 
  int     *new_centers_len;
  float **feature_extend;

  cl_mem d_feature;
  cl_mem d_membership;
  cl_mem d_cluster;

  cl_int err = 0;

  // 0th: initialize the timer at the beginning of the program
  

  // Create device buffers
  d_feature = clCreateBuffer(context, CL_MEM_READ_WRITE, npoints * COALESCE_SIZE * sizeof(float), NULL, &err );
  if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_feature (size:%d) => %d\n", npoints * COALESCE_SIZE, err);}
  d_membership = clCreateBuffer(context, CL_MEM_READ_WRITE, npoints * sizeof(int), NULL, &err );
  if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_membership (size:%d) => %d\n", npoints, err);}
  d_cluster = clCreateBuffer(context, CL_MEM_READ_WRITE, nclusters * nfeatures  * sizeof(float), NULL, &err );
  if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_cluster (size:%d) => %d\n", nclusters * nfeatures, err);}

  // 1st: time of buffer allocation
  //toc(&timer, "buffer allocation");

  prev_membership = (int*) malloc(npoints * sizeof(int));
  for (i=0; i<npoints; i++) {
    membership[i] = -1;
    prev_membership[i] = -1;
  }
  
  /* allocate space for and initialize new_centers_len and new_centers */
  new_centers_len = (int*) calloc(nclusters, sizeof(int));

  new_centers    = (float**) malloc(nclusters *            sizeof(float*));
  new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
  for (i=1; i<nclusters; i++)
      new_centers[i] = new_centers[i-1] + nfeatures;

  feature_extend    = (float**) malloc(npoints *  sizeof(float*));
  feature_extend[0] = (float*) calloc(npoints * COALESCE_SIZE, sizeof(float));
  for (i=1; i<npoints; i++)
      feature_extend[i] = feature_extend[i-1] + COALESCE_SIZE;

  for (i = 0; i < npoints; i++)
  {
      for (j = 0; j < nfeatures; j++)
      {
          feature_extend[i][j] += feature[i][j];
      }
  }

  // Write our data set into device buffers  
  //
  err = clEnqueueWriteBuffer(commands, d_feature, 1, 0, npoints * COALESCE_SIZE * sizeof(float), feature_extend[0], 0, 0, 0);
  if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_feature (size:%d) => %d\n", npoints * COALESCE_SIZE, err);}
  // err = clEnqueueWriteBuffer(commands, d_membership, 1, 0, npoints * sizeof(int), membership, 0, 0, 0);
  // if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_membership (size:%d) => %d\n", npoints, err);}

    timespec kernel_sum, kernel_diff;
    kernel_sum.tv_sec = 0;
    kernel_sum.tv_nsec = 0;

  do {
    timespec timer = tic();
     err = clEnqueueWriteBuffer(commands, d_cluster, 1, 0, nclusters * nfeatures * sizeof(float), clusters[0], 0, 0, 0);
    if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_cluster (size:%d) => %d\n", nclusters * nfeatures, err);}
      
    // 2nd: time of pageable-pinned memory copy
    toc(&timer, "memory copy");

    // Set the arguments to our compute kernel
    err = clSetKernelArg(kernel, 0, sizeof(void *), (void*) &d_feature);
    err |= clSetKernelArg(kernel, 1, sizeof(void *), (void*) &d_membership);
    err |= clSetKernelArg(kernel, 2, sizeof(void *), (void*) &d_cluster);

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

    clFinish(commands);

    // 4th: time of kernel execution
      toc(&timer, "kernel execution");

    err = clEnqueueReadBuffer(commands, d_membership, 1, 0, npoints * sizeof(int), membership, 0, 0, 0);
    if(err != CL_SUCCESS) { printf("ERROR: Memcopy Out\n"); }
    
    // 5th: time of data retrieving (PCIe + memcpy)
    toc(&timer, "data retrieving");

    delta = 0;
    for (i = 0; i < npoints; i++)
    {
        int cluster_id = membership[i];
        new_centers_len[cluster_id]++;
        if (membership[i] != prev_membership[i])
        {
            delta++;
            prev_membership[i] = membership[i];
        }

        for (j = 0; j < nfeatures; j++)
        {
            new_centers[cluster_id][j] += feature[i][j];
        }
    }
                
    /* replace old cluster centers with new_centers */
    /* CPU side of reduction */
    for (i=0; i<nclusters; i++) {
        for (j=0; j<nfeatures; j++) {
            if (new_centers_len[i] > 0)
                clusters[i][j] = new_centers[i][j] / new_centers_len[i];    /* take average i.e. sum/n */
            new_centers[i][j] = 0.0;    /* set back to 0 */
        }
        new_centers_len[i] = 0;         /* set back to 0 */
    } 
  } while (delta > threshold);

    printTimeSpec( kernel_sum, "Total kernel time" );
  free(prev_membership);
  clReleaseMemObject(d_feature);
  clReleaseMemObject(d_membership);
  clReleaseMemObject(d_cluster);
}

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
    int i, j;
    struct bench_args_t *args = (struct bench_args_t *)vargs;

    setup(args, context, commands, program, kernel);
//    float **feature    = (float**) malloc(819200 *             sizeof(float*));
//    feature[0] = (float*)  calloc(819200 * 34, sizeof(float));
//    for (i=1; i<819200; i++)
//        feature[i] = feature[i-1] + 34;
//
//    float  **clusters;					/* out: [nclusters][nfeatures] */
//
//    /* allocate space for returning variable clusters[] */
//    clusters    = (float**) malloc(5 *             sizeof(float*));
//    clusters[0] = (float*)  malloc(5 * 34 * sizeof(float));
//    for (i=1; i<5; i++)
//        clusters[i] = clusters[i-1] + 34;
//
//    /* randomly pick cluster centers */
//    for (i=0; i<5; i++) {
//        for (j=0; j<34; j++)
//            clusters[i][j] = feature[i][j];
//    }
//
//    int *membership = (int*) malloc(819200 * sizeof(int));
//
//    kmeansFPGA(feature, 34, 819200, 5, 0.1, clusters, membership, context, commands, program, kernel);
//    free(membership);
//    free(clusters[0]);
//    free(clusters);
//    free(feature[0]);
//    free(feature);
}

/* Input format:
%%: Section 1
uint8_t[32]: key
%%: Section 2
uint8_t[16]: input-text
*/

void input_to_data(int fd, void *vdata) {
    struct bench_args_t *data = (struct bench_args_t *)vdata;

    // Zero-out everything.
    memset(vdata,0,sizeof(struct bench_args_t));
    data->fd = fd;
}

void data_to_input(int fd, void *vdata) {

}

/* Output format:
%% Section 1
uint8_t[16]: output-text
*/

void output_to_data(int fd, void *vdata) {

}

void data_to_output(int fd, void *vdata) {

}

int check_data( void *vdata, void *vref ) {
  int has_errors = 0;

  // Return true if it's correct.
  return !has_errors;
}
