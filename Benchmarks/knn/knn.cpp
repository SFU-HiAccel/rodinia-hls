#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>
#include <sys/time.h>
#include <time.h>

typedef struct timespec timespec;
inline timespec diff(timespec start, timespec end)
{   
  timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}

inline timespec sum(timespec t1, timespec t2) {
  timespec temp;
  if (t1.tv_nsec + t2.tv_nsec >= 1000000000) {
    temp.tv_sec = t1.tv_sec + t2.tv_sec + 1;
    temp.tv_nsec = t1.tv_nsec + t2.tv_nsec - 1000000000;
  } else {
    temp.tv_sec = t1.tv_sec + t2.tv_sec;
    temp.tv_nsec = t1.tv_nsec + t2.tv_nsec;
  }
  return temp;
}

inline void printTimeSpec(timespec t, const char* prefix) {
  printf("%s: %d.%09d\n", prefix, (int)t.tv_sec, (int)t.tv_nsec);
}

inline timespec tic( )
{
    timespec start_time;
    clock_gettime(CLOCK_REALTIME, &start_time);
    return start_time;
}

inline void toc(timespec* start_time, timespec* timer)
{
    timespec current_time;
    clock_gettime(CLOCK_REALTIME, &current_time);
    *timer = sum(*timer, diff( *start_time, current_time ));
}




#define BATCH_OFFSET 1
#define BATCH_SIZE ( (BATCH_OFFSET) << 21 )

#define REC_LENGTH 49   // size of a record in db
#define REC_WINDOW 10   // number of records to read at a time
#define LATITUDE_POS 28 // location of latitude coordinates in input record
#define DBNAME_SIZE 64
#define OPEN 10000              // initial value of nearest neighbors, very large

struct db {
    int idx;
    char name[DBNAME_SIZE];
};



int binary_search(struct db* array, int n, int search) {
    int first = 0;
    int last = n - 1;
    int middle = (first + last) / 2;
 
    while (first <= last) {
        if (array[middle].idx <= search) {
            if( middle == n - 1 )
                return -1;
            else if( array[middle + 1].idx > search ) 
                return middle + 1;
            else 
                first = middle + 1;
        }
        else {
            if( middle == 0 )
                return 0;
            else if( array[middle - 1].idx <= search ) 
                return middle;
            else 
                last = middle - 1;
        }
 
        middle = (first + last) / 2;
    }

    return -1;
}


void write_result(struct db*  db_list, int* n_idx, float* n_dist, int k, int db_count) {

    fprintf(stderr, "The %d nearest neighbors are:\n", k);
    char record[REC_LENGTH];
    int i;
    for( i = 0 ; i < k ; i++ ) {
        if( n_dist[i] != OPEN ) {

            int db_idx = binary_search(db_list, db_count, n_idx[i]);    //find out which database is the record in
            if (db_idx == -1) {
                free(db_list);
                printf("can't find index\n");
                exit(0);
            }

            FILE *fp = fopen(db_list[db_idx].name, "r");
            if(!fp) {
                free(db_list);
                printf("error opening db %d\n", db_count);
                exit(1);
            }

            int rec_idx = n_idx[i] + 1;                               //read the record from the database  
            if (db_idx != 0)
                rec_idx = rec_idx - db_list[db_idx - 1].idx;
            while( rec_idx ){
                fgets(record, REC_LENGTH, fp);
                if (strcmp(record, "\n"))
                    rec_idx--;
            }
            fclose(fp);
            fprintf(stderr, "%s --> %f\n", record, n_dist[i]);
        }
    }

    return;
}


void update_neighbor_list(
    int k,
    int local_size,
    float* distance,
    float* n_dist,
    int* n_idx,
    float* max_dist,
    int* max_idx,
    int idx_base
)
{

    int m, j;
    for( m = 0; m < local_size; m++ ) {

        // compare each record with max value to find the nearest neighbors
        if( distance[m] < *max_dist ) {
            n_idx[ *max_idx ] = idx_base + m;
            n_dist[ *max_idx ] = distance[m];

            //update the max distance in the neighborhood
            *max_dist = -1;                          
            for( j = 0; j < k; j++ ) {      
                if( n_dist[j] > *max_dist ) {
                    *max_dist = n_dist[j];
                    *max_idx = j;
                }
            }
        } 
    }

    return;
}


void check_err(cl_int err, char* message) {
    if (err != CL_SUCCESS){
        printf("Error: Failed to %s!\n", message);
        printf("Test failed\n");
        exit(1);
    }
}


void print_profile(timespec* timers) {
    printf("\nProfile:\n");
    printTimeSpec(timers[0], "Create buffer");
    printTimeSpec(timers[1], "Read records from disk");
    printTimeSpec(timers[2], "Pre-process records");
    printTimeSpec(timers[3], "Write to buffer");
    printTimeSpec(timers[4], "Set arguments");
    printTimeSpec(timers[5], "Execute kernel");
    printTimeSpec(timers[6], "Raed from buffer");
    printTimeSpec(timers[7], "Update neighbor list");
    printTimeSpec(timers[8], "Release buffer");
    printTimeSpec(timers[9], "Write Result");
    return;
}



void launch_batch(
    char* batch,
    float* target,
    float* lat,
    float* lon,
    float* distance,
    int batch_size,
    int k,
    int* n_idx,
    float* n_dist,
    int batch_count,
    float* max_dist,
    int* max_idx,
    cl_context& context,
    cl_command_queue& commands, 
    cl_program& program, 
    cl_kernel& kernel,
    timespec* timers
) 
{ 
    timespec start_time = tic();
    int i;
    char* rec_iter;
    for( i = 0; i < batch_size; i++ ) {
        rec_iter = batch + ( i * REC_LENGTH + LATITUDE_POS - 1 );
        lat[i] = atof( rec_iter );
        lon[i] = atof( rec_iter + 5 );
    }
    toc(&start_time, timers + 2);

    start_time = tic();
    cl_int err;
    cl_mem lat_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, BATCH_SIZE * sizeof(float), NULL, &err);
    cl_mem lon_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, BATCH_SIZE * sizeof(float), NULL, &err);
    cl_mem distance_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, BATCH_SIZE * sizeof(float), NULL, &err);
    check_err(err, "allocate devise memory");
    toc(&start_time, timers);

    start_time = tic();
    err |= clEnqueueWriteBuffer(commands, lat_buffer, CL_TRUE, 0, BATCH_SIZE * sizeof(float), lat, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, lon_buffer, CL_TRUE, 0, BATCH_SIZE * sizeof(float), lon, 0, NULL, NULL);
    check_err(err, "write input buffer");
    toc(&start_time, timers + 3);

    start_time = tic();
    err = clSetKernelArg(kernel, 0, sizeof(float), target);
    err |= clSetKernelArg(kernel, 1, sizeof(float), target + 1);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &lat_buffer);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &lon_buffer);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &distance_buffer);
    check_err(err, "set argument");
    toc(&start_time, timers + 4);

    start_time = tic();
#ifdef C_KERNEL
    err = clEnqueueTask(commands, kernel, 0, NULL, NULL);
#else
    printf("Error: OpenCL kernel is not currently supported!\n");
    exit(1);
#endif
    check_err(err, "execute kernel");
    clFinish(commands);
    toc(&start_time, timers + 5);

    // Read back the results from the device to verify the output
    start_time = tic();
    err = clEnqueueReadBuffer( commands, distance_buffer, CL_TRUE, 0, batch_size * sizeof(float), distance, 0, NULL, NULL );  
    check_err(err, "read output buffer");
    toc(&start_time, timers + 6);


    start_time = tic();
    err = clReleaseMemObject(lat_buffer);
    err |= clReleaseMemObject(lon_buffer);
    err |= clReleaseMemObject(distance_buffer);
    check_err(err, "release memory object");
    toc(&start_time, timers + 8);

    start_time = tic();
    update_neighbor_list(
        k,
        batch_size,
        distance,
        n_dist,
        n_idx,
        max_dist,
        max_idx,
        batch_count * BATCH_SIZE
    );
    toc(&start_time, timers + 7);



    return;
}



void run_benchmark(FILE *flist, int k, float* target, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel){


    int* n_idx = (int *)malloc(k * sizeof(int));                       //initialize the nearest neighbor list    
    memset( n_idx, 0, k * sizeof( int ) );          
    float* n_dist = (float *)malloc(k * sizeof(float));
    int i;
    for ( i = 0; i < k; i++ ) 
        n_dist[i] = OPEN;


    int db_count = 0;
    int db_list_size = 1000;
    struct db* db_list = ( struct db* )malloc( db_list_size * sizeof( struct db ) );
    if(db_list == NULL) {
        fclose(flist);
        printf("error allocating db buffer\n");
        exit(1);
    }

    char dbname[DBNAME_SIZE];
    if(fscanf(flist, "%s\n", dbname) != 1) {
        fclose(flist);
        fprintf(stderr, "error reading filelist\n");
        exit(1);
    }
    FILE *fp;
    fp = fopen(dbname, "r");
    if(!fp) {
        fclose(flist);
        printf("error opening db 1\n");
        exit(1);
    }


    int rec_count = 0, global_rec_count = 0, batch_pointer = 0, batch_count = 0;

    char* batch = (char *)malloc(REC_LENGTH * BATCH_SIZE);
    float* lat = (float *)malloc(sizeof(float) * BATCH_SIZE);
    memset(lat, 0, sizeof(float) * BATCH_SIZE);
    float* lon = (float *)malloc(sizeof(float) * BATCH_SIZE);
    memset(lon, 0, sizeof(float) * BATCH_SIZE);
    float* distance = (float *)malloc(sizeof(float) * BATCH_SIZE);


    float max_dist = OPEN;
    int max_idx = 0;

    timespec timers[10];
    for(i = 0; i < 10; i++) {
        timers[i].tv_sec = 0;
        timers[i].tv_nsec = 0;
    }
    timespec start_time;

    int done = 0;
    while(!done) {

        start_time = tic();
        int batch_window = BATCH_SIZE - batch_pointer;               
        /*
        batch_window is the maximum number of records that can be read in one time
        It depends on how many records are already in the batch
        */

        rec_count = fread(
            batch + batch_pointer * REC_LENGTH, 
            REC_LENGTH, batch_window, 
            fp
        );                                                           //rec_count is the number of records readed each time
        batch_pointer += rec_count;                                  //batch_pointer is the number of records already in the batch
        global_rec_count += rec_count;                               //global_rec_count is the total number of records
        toc(&start_time, timers + 1);

        if ( batch_pointer == BATCH_SIZE ) {                         //a batch is full, ready to be dispatched for computation
            launch_batch(
                batch,
                target,
                lat,
                lon,
                distance,
                batch_pointer,
                k,
                n_idx,
                n_dist,
                batch_count,
                &max_dist,
                &max_idx,
                context,
                commands,
                program,
                kernel,
                timers
            );
            batch_count++;                                           //number of batches increases
            batch_pointer = 0;                                       //now batch is empty
        }

        if( rec_count != batch_window ) {
            if(!ferror(flist)) {                                     //eof of db
                fclose(fp);

                //save db information
                db_list[db_count].idx = global_rec_count;            //db_list is the partial sum of records of each db
                memcpy(db_list[db_count].name, dbname, DBNAME_SIZE * sizeof (char) );   
                db_count++;

                if(db_count == db_list_size) {                       //too many db   
                    db_list = (struct db* )realloc(db_list, db_list_size * 2 * sizeof( struct db ));
                    if(db_list == NULL) {
                        fclose(flist);
                        printf("error allocating db buffer\n");
                        exit(1);
                    }
                    db_list_size *= 2;
                }

                if(feof(flist)){                                     //eof of filelist

                    if( batch_pointer != 0 ) {                       //there are still records in the batch not computed
                        launch_batch(
                            batch,
                            target,
                            lat,
                            lon,
                            distance,
                            batch_pointer,
                            k,
                            n_idx,
                            n_dist,
                            batch_count,
                            &max_dist,
                            &max_idx,
                            context,
                            commands,
                            program,
                            kernel,
                            timers
                        );
                        batch_count++;
                        batch_pointer = 0;
                    }

                    fclose(flist);
                    done = 1;                                        //exit loop
                }

                else {                                               //filelist is not empty yet               

                    if(fscanf(flist, "%s\n", dbname) != 1) {         //read new db
                        fclose(flist);
                        free(db_list);
                        fprintf(stderr, "error reading filelist\n");
                        exit(1);
                    }

                    fp = fopen(dbname, "r");
                    if(!fp) {
                        fclose(flist);
                        free(db_list);
                        printf("error opening db %d\n", db_count);
                        exit(1);
                    }
                }
            }
            else{                                                   //flist error
                fclose(fp);
                free(db_list);
                perror("Error");
                exit(1);
            }
        }      
    }   //exit while loop

    start_time = tic();
    write_result(db_list, n_idx, n_dist, k, db_count);
    toc(&start_time, timers + 9);

#if defined (PROFILE)
    print_profile(timers);
#endif

    free(n_dist);
    free(n_idx);
    free(db_list);
    free(batch);
    free(lat);
    free(lon);
    free(distance);
    return;
}

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

int main( int argc, char* argv[] ) {

    if( argc != 6 ) {
        fprintf(stderr, "Invalid set of arguments\n");
        exit(-1);
    }

    FILE *flist;
    flist = fopen(argv[1], "r");
    if(!flist) {
        printf("error opening flist\n");
        exit(1);
    }

    int k = atoi(argv[2]);

    float target[2] = { atof(argv[3]), atof(argv[4]) };

    cl_platform_id platform_id;         // platform id
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
   
    char cl_platform_vendor[1001];
    char cl_platform_name[1001];

    cl_int err;

    err = clGetPlatformIDs(1,&platform_id,NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to find an OpenCL platform!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    err = clGetPlatformInfo(platform_id,CL_PLATFORM_VENDOR,1000,(void *)cl_platform_vendor,NULL);
    if (err != CL_SUCCESS){
        printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("CL_PLATFORM_VENDOR %s\n",cl_platform_vendor);
    err = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME,1000,(void *)cl_platform_name,NULL);
    if (err != CL_SUCCESS){
        printf("Error: clGetPlatformInfo(CL_PLATFORM_NAME) failed!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("CL_PLATFORM_NAME %s\n",cl_platform_name);
 
    // Connect to a compute device
    int fpga = 0;
#if defined (FPGA_DEVICE)
    fpga = 1;
#endif
    err = clGetDeviceIDs(platform_id, fpga ? CL_DEVICE_TYPE_ACCELERATOR : CL_DEVICE_TYPE_CPU,
        1, &device_id, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to create a device group!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
  
    // Create a compute context 
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context){
        printf("Error: Failed to create a compute context!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands){
        printf("Error: Failed to create a command commands!\n");
        printf("Error: code %i\n",err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    int status;

    // Create Program Objects
    //
  
    // Load binary from disk
    unsigned char *kernelbinary;
    // The sixth parameter specifies the name of the kernel binary
    char *xclbin=argv[5];
    printf("loading %s\n", xclbin);
    int n_i = load_file_to_memory(xclbin, (char **) &kernelbinary);
    if (n_i < 0) {
        printf("failed to load kernel from xclbin: %s\n", xclbin);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    size_t n = n_i;
    // Create the compute program from offline
    program = clCreateProgramWithBinary(context, 1, &device_id, &n,
        (const unsigned char **) &kernelbinary, &status, &err);
    if ((!program) || (err!=CL_SUCCESS)) {
        printf("Error: Failed to create compute program from binary %d!\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS){
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "workload", &err);
    if (!kernel || err != CL_SUCCESS){
        printf("Error: Failed to create compute kernel!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }


    run_benchmark( flist, k, target, context, commands, program, kernel );

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}
