/*
*   Byte-oriented AES-256 implementation.
*   All lookup tables replaced with 'on the fly' calculations.
*/
#ifndef KMEANS_H
#define KMEANS_H
#include <inttypes.h>
//#include <CL/cl.h>

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

//typedef struct {
//	cl_context context;
//	cl_command_queue cmd_queue;
//	cl_program program;
//	cl_kernel kernel;
//} cl_param_t;

//int setup(struct bench_args_t *args, cl_param_t *cl_param);
//
///* cluster.c */
//int cluster(int      numObjects,      /* number of input objects */
//            int      numAttributes,   /* size of attribute of each object */
//            float  **attributes,      /* [numObjects][numAttributes] */
//            int      nclusters,
//            float    threshold,       /* in:   */
//            float ***cluster_centres, /* out: [best_nclusters][numAttributes] */
//            cl_param_t *cl_param
//            );
//
///* kmeans_clustering.c */
//float** kmeans_clustering(float **feature,    /* in: [npoints][nfeatures] */
//                          int     nfeatures,
//                          int     npoints,
//                          int     nclusters,
//                          float   threshold,
//                          int *membership,
//                          cl_param_t *cl_param);

//void kmeansFPGA(float **feature,    /* in: [npoints][nfeatures] */
//              int     n_features,  /* in */
//              int     n_points,/* in */
//              int     n_clusters,/* in */
//              float   threshold,
//              float **clusters,   /* out */
//              int *membership,
//              cl_param_t *cl_param);
////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  int fd;
};
#endif
