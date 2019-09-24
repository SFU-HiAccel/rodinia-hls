#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/types.h>
#include <fcntl.h>
#include <CL/opencl.h>
#include "kmeans.h"
extern void kmeansFPGA(float **feature,    /* in: [npoints][nfeatures] */
                       int     nfeatures,  /* in */
                       int     npoints,/* in */
                       int     nclusters,/* in */
                       float   threshold,
                       float **clusters,   /* out */
                       int *membership,
                       cl_context& context,
                       cl_command_queue& commands,
                       cl_program& program,
                       cl_kernel& kernel);

/*----< kmeans_clustering() >---------------------------------------------*/
float** kmeans_clustering(float **feature,    /* in: [npoints][nfeatures] */
                          int     nfeatures,
                          int     npoints,
                          int     nclusters,
                          float   threshold,
                          int *membership,
                          cl_context& context,
                          cl_command_queue& commands,
                          cl_program& program,
                          cl_kernel& kernel)
{

    int      i, j;
    float  **clusters;					/* out: [nclusters][nfeatures] */

    /* allocate space for returning variable clusters[] */
    clusters    = (float**) malloc(nclusters *             sizeof(float*));
    clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
    for (i=1; i<nclusters; i++)
        clusters[i] = clusters[i-1] + nfeatures;

    /* randomly pick cluster centers */
    for (i=0; i<nclusters; i++) {
        for (j=0; j<nfeatures; j++)
            clusters[i][j] = feature[i][j];
    }

    // for (i=0; i<npoints; i++)
    //     membership[i] = -1;

    // FPGA
    kmeansFPGA(feature,         /* in: [npoints][nfeatures] */
               nfeatures,       /* number of attributes for each point */
               npoints,         /* number of data points */
               nclusters,       /* number of clusters */
               threshold,
               clusters,        /* out: [nclusters][nfeatures] */
               membership,
               context, commands, program, kernel
    );

    return clusters;
}

int cluster(int      numObjects,      /* number of input objects */
            int      numAttributes,   /* size of attribute of each object */
            float  **attributes,      /* [numObjects][numAttributes] */
            int      nclusters,
            float    threshold,       /* in:   */
            float ***cluster_centres, /* out: [best_nclusters][numAttributes] */
            cl_context& context,
            cl_command_queue& commands,
            cl_program& program,
            cl_kernel& kernel)
{
    int    *membership;
    float **tmp_cluster_centres;

    membership = (int*) malloc(numObjects * sizeof(int));

    srand(7);
    /* perform regular Kmeans */
    tmp_cluster_centres = kmeans_clustering(attributes,
                                            numAttributes,
                                            numObjects,
                                            nclusters,
                                            threshold,
                                            membership,
                                            context, commands, program, kernel);

    if (*cluster_centres) {
        free((*cluster_centres)[0]);
        free(*cluster_centres);
    }
    *cluster_centres = tmp_cluster_centres;

    free(membership);

    return 0;
}

int setup(struct bench_args_t *args, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel) {
           int     nclusters=5;         
           float  *buf;
           float **attributes;
           float **cluster_centres=NULL;
           int     i, j;
                
           int     numAttributes;
           int     numObjects;        
           char    line[1024];
           int     nloops = 1;
           float   threshold = 0.001;

    numAttributes = 0;
    numObjects = 0;

    /* from the input file, get the numAttributes and numObjects ------------*/
    FILE *infile;
    FILE *outfile;
    if ((infile = fdopen(args->fd, "r")) == NULL) {
        fprintf(stderr, "Error: no such file (%d)\n", args->fd);
        exit(1);
    }
    while (fgets(line, 1024, infile) != NULL)
        if (strtok(line, " \t\n") != 0)
            numObjects++;
    rewind(infile);
    while (fgets(line, 1024, infile) != NULL) {
        if (strtok(line, " \t\n") != 0) {
            /* ignore the id (first attribute): numAttributes = 1; */
            while (strtok(NULL, " ,\t\n") != NULL) numAttributes++;
            break;
        }
    }

    printf("number of clusters %d\n",nclusters);
    printf("number of attributes %d\n",numAttributes);
    printf("number of objects %d\n\n",numObjects);

    /* allocate space for attributes[] and read attributes of all objects */
    buf           = (float*) malloc(numObjects*numAttributes*sizeof(float));
    attributes    = (float**)malloc(numObjects*             sizeof(float*));
    attributes[0] = (float*) malloc(numObjects*numAttributes*sizeof(float));
    for (i=1; i<numObjects; i++)
        attributes[i] = attributes[i-1] + numAttributes;
    rewind(infile);
    i = 0;
    while (fgets(line, 1024, infile) != NULL) {
        if (strtok(line, " \t\n") == NULL) continue; 
        for (j=0; j<numAttributes; j++) {
            buf[i] = atof(strtok(NULL, " ,\t\n"));
            i++;
        }
    }
    fclose(infile);
   
    printf("I/O completed\n");  

    memcpy(attributes[0], buf, numObjects*numAttributes*sizeof(float));

    //timing = omp_get_wtime();
    for (i=0; i<nloops; i++) {
        
        cluster_centres = NULL;
        cluster(numObjects,
                numAttributes,
                attributes,           /* [numObjects][numAttributes] */                
                nclusters,
                threshold,
                &cluster_centres,
                context, commands, program, kernel);
     
    }
    //timing = omp_get_wtime() - timing;

    if ((outfile = fopen("out.data", "w")) == NULL) {
        fprintf(stderr, "Error: no such file (out.data)\n");
        exit(1);
    }
    for (i=0; i< nclusters; i++) {
//        printf("%d: ", i);
        for (j=0; j<numAttributes; j++) {
            fprintf(outfile, "%.6f ", cluster_centres[i][j]);
            printf("%.6f ", cluster_centres[i][j]);
        }
        fprintf(outfile, "\n");
        printf("\n");
    }
    fclose(outfile);
    //printf("Time for process: %f\n", timing);

    free(attributes);
    free(cluster_centres[0]);
    free(cluster_centres);
    free(buf);
    return(0);
}