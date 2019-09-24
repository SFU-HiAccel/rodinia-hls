//Copyright (c) 2006-2009 Princeton University
//All rights reserved.

//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions are met:
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//    * Neither the name of Princeton University nor the
//      names of its contributors may be used to endorse or promote products
//      derived from this software without specific prior written permission.

//THIS SOFTWARE IS PROVIDED BY PRINCETON UNIVERSITY ``AS IS'' AND ANY
//EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL PRINCETON UNIVERSITY BE LIABLE FOR ANY
//DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.




#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <sys/resource.h>
#include <limits.h>

#include "_cl_helper.h"


using namespace std;

#define MAXNAMESIZE 1024 // max filename length
#define SEED 1
/* increase this to reduce probability of random error */
/* increasing it also ups running time of "speedy" part of the code */
/* SP = 1 seems to be fine */
#define SP 1 // number of repetitions of speedy must be >=1

/* higher ITER --> more likely to get correct # of centers */
/* higher ITER also scales the running time almost linearly */
#define ITER 1 // iterate ITER* k log k times; ITER >= 1

//#define PRINTPROFILE


/* this structure represents a point */
/* these will be passed around to avoid copying coordinates */
typedef struct {
  float weight;
  float *coord;
  int assign;  /* index of point where this one is assigned */
  float cost;  /* cost of that assignment, weight*distance */
} Point;

/* this is the array of points */
typedef struct {
  int num; /* number of points; may not be N if this is a sample */
  int dim;  /* dimensionality */
  Point *p; /* the array itself */
} Points;

char *switch_membership; //whether to switch membership in pgain
bool* is_center; //whether a point is a center
int* center_table; //index table of centers

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return (double)t.tv_sec+t.tv_usec*1e-6;
}

int isIdentical(float *i, float *j, int D)
// tells whether two points of D dimensions are identical
{
  int a = 0;
  int equal = 1;

  while (equal && a < D) {
    if (i[a] != j[a]) equal = 0;
    else a++;
  }
  if (equal) return 1;
  else return 0;

}

/* comparator for floating point numbers */
int floatcomp(const void *i, const void *j)
{
  float a, b;
  a = *(float *)(i);
  b = *(float *)(j);
  if (a > b) return (1);
  if (a < b) return (-1);
  return(0);
}

/* shuffle points into random order */
void shuffle(Points *points)
{

  int i, j;
  Point temp;
  for (i=0;i<points->num-1;i++) {
    j=(lrand48()%(points->num - i)) + i;
    temp = points->p[i];
    points->p[i] = points->p[j];
    points->p[j] = temp;
  }

}

/* shuffle an array of integers */
void intshuffle(int *intarray, int length)
{

  int i, j;
  int temp;
  for (i=0;i<length;i++) {
    j=(lrand48()%(length - i))+i;
    temp = intarray[i];
    intarray[i]=intarray[j];
    intarray[j]=temp;
  }

}


/* compute Euclidean distance squared between two points */
float dist(Point p1, Point p2, int dim)
{
  int i;
  float result=0.0;
  for (i=0;i<dim;i++)
    result += (p1.coord[i] - p2.coord[i])*(p1.coord[i] - p2.coord[i]);

  return(result);
}

/* run speedy on the points, return total cost of solution */
float pspeedy(Points *points, float z, int *kcenter)
{
  int i, k;

  /* create center at first point, send it to itself */
  for( k = 0; k < points->num; k++ )    {
    float distance = dist(points->p[k],points->p[0],points->dim);
    points->p[k].cost = distance * points->p[k].weight;
    points->p[k].assign=0;
  }

    *kcenter = 1;
    

  for(i = 1; i < points->num; i++ )  {
    bool to_open = ((float)lrand48()/(float)INT_MAX)<(points->p[i].cost/z);
    if( to_open )  {
	     (*kcenter)++;

      for( k = 0; k < points->num; k++ )  {
        float distance = dist(points->p[i],points->p[k],points->dim);
        if( distance*points->p[k].weight < points->p[k].cost )  {
          points->p[k].cost = distance * points->p[k].weight;
          points->p[k].assign=i;
        }
      }
    }
  }

  double mytotal = 0;
  for( int k = 0; k < points->num; k++ )  {
    mytotal += points->p[k].cost;
  }

  return(z * (*kcenter) + mytotal);
}


/* For a given point x, find the cost of the following operation:
 * -- open a facility at x if there isn't already one there,
 * -- for points y such that the assignment distance of y exceeds dist(y, x),
 *    make y a member of x,
 * -- for facilities y such that reassigning y and all its members to x 
 *    would save cost, realize this closing and reassignment.
 * 
 * If the cost of this operation is negative (i.e., if this entire operation
 * saves cost), perform this operation and return the amount of cost saved;
 * otherwise, do nothing.
 */

/* numcenters will be updated to reflect the new number of centers */
/* z is the facility cost, x is the number of this point in the array 
   points */

double pgain(int x, Points *points, double z, int *numcenters, timespec* timers)
{
  if(*numcenters > MAX_WORK_MEM_SIZE) {
    printf("number of centers too big %d\n", *numcenters);
    exit(1);
  }
  timespec start_time = tic();

  int i, j;

  int count = 0;
  for(i = 0; i < points->num; i++) {
    if( is_center[i] ) {
      center_table[i] = count++;
    }
  }

  float* coord = (float *)calloc(BATCH_SIZE * DIM, sizeof(float));
  for(i = 0; i < points->num; i++){
    for(j = 0; j < points->dim; j++){
      coord[i * DIM  + j] = points->p[i].coord[j];
    }
  }

  float* weight = (float *)calloc(BATCH_SIZE, sizeof(float));
  for(i = 0; i < points->num; i++){
    weight[i] = points->p[i].weight;
  }

  float target[DIM];
  memset(target, 0, DIM * sizeof(float));
  for(i = 0; i < points->dim; i++)
    target[i] = points->p[x].coord[i];

  float *cost = (float*)malloc(BATCH_SIZE * sizeof(float));
  for(i = 0; i < points->num; i++){
    cost[i] = points->p[i].cost;
  }

  int *assign = (int*)malloc(BATCH_SIZE * sizeof(int));
  for(i = 0; i < points->num; i++){
    assign[i] = points->p[i].assign;
  }

  float *work_mem = (float *)malloc(*numcenters * sizeof(float));
  toc(&start_time, timers);

  double cost_of_opening_x = _clTask(coord, weight, target, cost, assign, center_table, switch_membership, work_mem, points->num, *numcenters, timers);
  start_time = tic();

  free(weight);
  free(coord);
  free(cost);
  free(assign);

  // at this time, we can calculate the cost of opening a center
  // at x; if it is negative, we'll go through with opening it
  int number_of_centers_to_close = 0;

  for ( int i = 0; i < points->num; i++ ) {
    if( is_center[i] ) {
      double low = z + work_mem[center_table[i]];
      work_mem[center_table[i]] = low;
      if ( low > 0 ) {
        // i is a median, and
        // if we were to open x (which we still may not) we'd close i
        // note, we'll ignore the following quantity unless we do open x

        ++number_of_centers_to_close;  
        cost_of_opening_x -= low;
      }
    }
  }

  cost_of_opening_x += z;

  // Now, check whether opening x would save cost; if so, do it, and
  // otherwise do nothing

  if ( cost_of_opening_x < 0 ) {
    //  we'd save money by opening x; we'll do it
    for ( int i = 0; i < points->num; i++ ) {
      bool close_center = work_mem[center_table[points->p[i].assign]] > 0 ;
      if ( switch_membership[i] || close_center ) {
        // Either i's median (which may be i itself) is closing,
        // or i is closer to x than to its current median
        points->p[i].cost = dist(points->p[i], points->p[x], points->dim) * points->p[i].weight;
        points->p[i].assign = x;
      }
    }
    for( int i = 0; i < points->num; i++ ) {
      if( is_center[i] && work_mem[center_table[i]] > 0 ) 
        is_center[i] = false;
    }
    if( x >= 0 && x < points->num ) 
      is_center[x] = true;

    *numcenters = *numcenters + 1 - number_of_centers_to_close;
    
  }
  else 
      cost_of_opening_x = 0;  // the value we'll return

  free(work_mem);  
  toc(&start_time, timers);
#ifdef PRINTPROFILE
  printf("\nProfile:\n");
    printTimeSpec(timers[0], "CPU");
    printTimeSpec(timers[1], "Write to buffer");
    printTimeSpec(timers[2], "Execute kernel");
    printTimeSpec(timers[3], "Read from buffer");
#endif
  return -cost_of_opening_x;
}


/* facility location on the points using local search */
/* z is the facility cost, returns the total cost and # of centers */
/* assumes we are seeded with a reasonable solution */
/* cost should represent this solution's cost */
/* halt if there is < e improvement after iter calls to gain */
/* feasible is an array of numfeasible points which may be centers */

float pFL(Points *points, int *feasible, int numfeasible,
	  float z, int *k, double cost, int iter, float e, timespec* timers)
{

  int i;
  int x;
  double change = cost;

  /* continue until we run iter iterations without improvement */
  /* stop instead if improvement is less than e */
  while (change / cost > 1.0 * e) {
    change = 0.0;
    /* randomize order in which centers are considered */

    intshuffle(feasible, numfeasible);

    for (i = 0; i < iter; i++) {
      x = i % numfeasible;
      change += pgain(feasible[x], points, z, k, timers);
    }

    cost -= change;

  }
  return(cost);
}

int selectfeasible_fast(Points *points, int **feasible, int kmin)
{


  int numfeasible = points->num;
  if (numfeasible > (ITER*kmin*log((double)kmin)))
    numfeasible = (int)(ITER*kmin*log((double)kmin));
  *feasible = (int *)malloc(numfeasible*sizeof(int));
  
  float* accumweight;
  float totalweight;

  /* 
     Calcuate my block. 
     For now this routine does not seem to be the bottleneck, so it is not parallelized. 
     When necessary, this can be parallelized by setting k1 and k2 to 
     proper values and calling this routine from all threads ( it is called only
     by thread 0 for now ). 
     Note that when parallelized, the randomization might not be the same and it might
     not be difficult to measure the parallel speed-up for the whole program. 
   */

  /* not many points, all will be feasible */
  if (numfeasible == points->num) {
    for (int i = 0; i < numfeasible; i++)
      (*feasible)[i] = i;
    return numfeasible;
  }

  accumweight= (float*)malloc(sizeof(float)*points->num);
  accumweight[0] = points->p[0].weight;
  totalweight=0;
  for( int i = 1; i < points->num; i++ ) {
    accumweight[i] = accumweight[i-1] + points->p[i].weight;
  }
  totalweight=accumweight[points->num-1];

  float w;
  int l,r,k;
  for(int i = 0; i < numfeasible; i++ ) {
    w = (lrand48()/(float)INT_MAX)*totalweight;
    //binary search
    l=0;
    r=points->num-1;
    if( accumweight[0] > w )  { 
      (*feasible)[i]=0; 
      continue;
    }
    while( l+1 < r ) {
      k = (l+r)/2;
      if( accumweight[k] > w ) 
        r = k;
      else 
        l=k;
    }
    (*feasible)[i]=r;
  }

  free(accumweight); 

  return numfeasible;
}

/* compute approximate kmedian on the points */
float pkmedian(Points *points, int kmin, int kmax, int* kfinal, timespec* timers)
{
  static int k;
  static int *feasible;
  static int numfeasible;

  double hiz = 0.0;
  int kk;
  for ( kk = 0;kk < points->num; kk++ ) 
    hiz += dist(points->p[kk], points->p[0],points->dim) * points->p[kk].weight;
  
  double cost;
  double z = hiz / 2.0;
  /* NEW: Check whether more centers than points! */
  if (points->num <= kmax) {
    /* just return all points as facilities */
    for (kk = 0; kk < points->num; kk++) {
      points->p[kk].assign = kk;
      points->p[kk].cost = 0;
    }
    cost = 0;
    *kfinal = k;
    return cost;
  }

  shuffle(points);
  cost = pspeedy(points, z, &k);


  int i = 0;
  /* give speedy SP chances to get at least kmin/2 facilities */
  while ((k < kmin)&&(i<SP)) {
    cost = pspeedy(points, z, &k);
    i++;
  }

  /* if still not enough facilities, assume z is too high */
  while (k < kmin) {
    if (i >= SP) {
      hiz = z; 
      z = hiz / 2.0; 
      i = 0;
    }
    shuffle(points);
    cost = pspeedy(points, z, &k);
    i++;
  }

  /* now we begin the binary search for real */
  /* must designate some points as feasible centers */
  /* this creates more consistancy between FL runs */
  /* helps to guarantee correct # of centers at the end */
  
  numfeasible = selectfeasible_fast(points,&feasible,kmin);
  for( int i = 0; i< points->num; i++ ) 
    is_center[points->p[i].assign]= true;

  double loz = 0.0;

  while(1) {

    /* first get a rough estimate on the FL solution */
    //    pthread_barrier_wait(barrier);

    cost = pFL(points, feasible, numfeasible,
	       z, &k, cost, (int)(ITER*kmax*log((double)kmax)), 0.1, timers);

    /* if number of centers seems good, try a more accurate FL */
    if (((k <= (1.1)*kmax)&&(k >= (0.9)*kmin))||((k <= kmax+2)&&(k >= kmin-2))) {

      /* may need to run a little inter here before halting without
	 improvement */
      cost = pFL(points, feasible, numfeasible,
		 z, &k, cost, (int)(ITER*kmax*log((double)kmax)), 0.001, timers);
    }

    if (k > kmax) {
      /* facilities too cheap */
      /* increase facility cost and up the cost accordingly */
      loz = z; z = (hiz+loz)/2.0;
      cost += (z-loz)*k;
    }
    if (k < kmin) {
      /* facilities too expensive */
      /* decrease facility cost and reduce the cost accordingly */
      hiz = z; z = (hiz+loz)/2.0;
      cost += (z-hiz)*k;
    }

    /* if k is good, return the result */
    /* if we're stuck, just give up and return what we have */
    if (((k <= kmax)&&(k >= kmin))||((loz >= (0.999)*hiz)) )
      break;
  }

  //clean up...
    free(feasible); 
    *kfinal = k;

  return cost;
}

/* compute the means for the k clusters */
int contcenters(Points *points)
{
  int i, ii;
  float relweight;

  for (i = 0; i < points->num; i++) {
    /* compute relative weight of this point to the cluster */
    if (points->p[i].assign != i) {
      relweight = points->p[points->p[i].assign].weight + points->p[i].weight;
      relweight = points->p[i].weight / relweight;
      for (ii=0;ii<points->dim;ii++) {
        points->p[points->p[i].assign].coord[ii] *= 1.0 - relweight;
        points->p[points->p[i].assign].coord[ii] += points->p[i].coord[ii] * relweight;
      }
      points->p[points->p[i].assign].weight += points->p[i].weight;
    }
  }
  
  return 0;
}

/* copy centers from points to centers */
void copycenters(Points *points, Points* centers, int* centerIDs, int offset)
{
  int i;
  bool *is_a_median = (bool *) calloc(points->num, sizeof(bool));

  /* mark the centers */
  for ( i = 0; i < points->num; i++ ) 
    is_a_median[points->p[i].assign] = 1;

  int k = centers->num;

  /* count how many  */
  for ( i = 0; i < points->num; i++ ) {
    if ( is_a_median[i] ) {
      memcpy( centers->p[k].coord, points->p[i].coord, points->dim * sizeof(float));
      centers->p[k].weight = points->p[i].weight;
      centerIDs[k] = i + offset;
      k++;
    }
  }

  centers->num = k;

  free(is_a_median);
}

class PStream {
public:
  virtual size_t read( float* dest, int dim, int num ) = 0;
  virtual int ferror() = 0;
  virtual int feof() = 0;
  virtual ~PStream() {
  }
};

//synthetic stream
class SimStream : public PStream {
public:
  SimStream(int n_ ) {
    n = n_;
  }
  size_t read( float* dest, int dim, int num ) {
    size_t count = 0;
    for( int i = 0; i < num && n > 0; i++ ) {
      for( int k = 0; k < dim; k++ ) {
	dest[i*dim + k] = lrand48()/(float)INT_MAX;
      }
      n--;
      count++;
    }
    return count;
  }
  int ferror() {
    return 0;
  }
  int feof() {
    return n <= 0;
  }
  ~SimStream() { 
  }
private:
  int n;
};

class FileStream : public PStream {
public:
  FileStream(char* filename) {
    fp = fopen( filename, "rb");
    if( fp == NULL ) {
      fprintf(stderr,"error opening file %s\n.",filename);
      exit(1);
    }
  }
  size_t read( float* dest, int dim, int num ) {
    return std::fread(dest, sizeof(float)*dim, num, fp); 
  }
  int ferror() {
    return std::ferror(fp);
  }
  int feof() {
    return std::feof(fp);
  }
  ~FileStream() {
    printf("closing file stream\n");
    fclose(fp);
  }
private:
  FILE* fp;
};

void outcenterIDs( Points* centers, int* centerIDs, char* outfile ) {
  FILE* fp = fopen(outfile, "w");
  if( fp==NULL ) {
    fprintf(stderr, "error opening %s\n",outfile);
    exit(1);
  }
  int* is_a_median = (int*)calloc( sizeof(int), centers->num );
  for( int i =0 ; i< centers->num; i++ ) {
    is_a_median[centers->p[i].assign] = 1;
  }

  for( int i = 0; i < centers->num; i++ ) {
    if( is_a_median[i] ) {
      fprintf(fp, "%u\n", (unsigned int)centerIDs[i]);
      fprintf(fp, "%lf\n", centers->p[i].weight);
      for( int k = 0; k < centers->dim; k++ ) 
        fprintf(fp, "%lf ", centers->p[i].coord[k]);
      fprintf(fp,"\n\n");
    }
  }
  fclose(fp);
}

void streamCluster( PStream* stream, 
		    int kmin, int kmax, int dim, int centersize, char* outfile )
{
  int i;
  timespec timers[10];
    for(i = 0; i < 10; i++) {
        timers[i].tv_sec = 0;
        timers[i].tv_nsec = 0;
    }
  float* block = (float*)malloc( BATCH_SIZE*dim*sizeof(float) );
  float* centerBlock = (float*)malloc(centersize*dim*sizeof(float) );
  int* centerIDs = (int*)malloc(centersize*dim*sizeof(int));

  if( block == NULL ) { 
    fprintf(stderr,"not enough memory for a chunk!\n");
    exit(1);
  }

  Points points;
  points.dim = dim;
  points.num = BATCH_SIZE;
  points.p = (Point *)malloc(BATCH_SIZE*sizeof(Point));
  for( int i = 0; i < BATCH_SIZE; i++ ) 
    points.p[i].coord = &block[i*dim];
  

  Points centers;
  centers.dim = dim;
  centers.p = (Point *)malloc(centersize*sizeof(Point));
  centers.num = 0;

  for( int i = 0; i< centersize; i++ ) {
    centers.p[i].coord = &centerBlock[i*dim];
    centers.p[i].weight = 1.0;
  }

  int IDoffset = 0;
  int kfinal;
  while(1) {

    size_t numRead  = stream->read(block, dim, BATCH_SIZE ); 
    fprintf(stderr,"read %d points\n",(int)numRead);

    if( stream->ferror() || (numRead < (unsigned int)BATCH_SIZE && !stream->feof()) ) {
      fprintf(stderr, "error reading data!\n");
      exit(1);
    }

    points.num = numRead;
    for( int i = 0; i < points.num; i++ ) {
      points.p[i].weight = 1.0;
    }

    switch_membership = (char*)malloc(BATCH_SIZE*sizeof(char));
    is_center = (bool*)calloc(points.num,sizeof(bool));
    center_table = (int*)malloc(BATCH_SIZE*sizeof(int));

    pkmedian(&points,kmin,kmax,&kfinal, timers);

    fprintf(stderr,"finish local search\n");
    contcenters(&points);
    if( kfinal + centers.num > centersize ) {
      //here we don't handle the situation where # of centers gets too large. 
      fprintf(stderr,"oops! no more space for centers\n");
      exit(1);
    }

    copycenters(&points, &centers, centerIDs, IDoffset);
    IDoffset += numRead;

    free(is_center);
    free(switch_membership);
    free(center_table);

    if( stream->feof() ) {
      break;
    }
  }

  //finally cluster all temp centers
  if(centers.num > BATCH_SIZE) {
    printf("ERROR: number of intermediate centers is too large!\n");
    _clClean();
    exit(1);
  }
  else{
    switch_membership = (char*)malloc(BATCH_SIZE*sizeof(char));
    is_center = (bool*)calloc(centers.num,sizeof(bool));
    center_table = (int*)malloc(BATCH_SIZE*sizeof(int));

    pkmedian(&centers, kmin, kmax ,&kfinal, timers);
    contcenters(&centers);
    outcenterIDs(&centers, centerIDs, outfile);

    free(is_center);
    free(switch_membership);
    free(center_table);
  }
}

int main(int argc, char **argv)
{



	fflush(NULL);


  if (argc<9) {
    fprintf(stderr,"usage: %s k1 k2 d n chunksize clustersize infile outfile nproc\n",
	    argv[0]);
    fprintf(stderr,"  k1:          Min. number of centers allowed\n");
    fprintf(stderr,"  k2:          Max. number of centers allowed\n");
    fprintf(stderr,"  d:           Dimension of each data point\n");
    fprintf(stderr,"  n:           Number of data points\n");
    fprintf(stderr,"  clustersize: Maximum number of intermediate centers\n");
    fprintf(stderr,"  infile:      Input file (if n<=0)\n");
    fprintf(stderr,"  outfile:     Output file\n");
    fprintf(stderr,"  bitstream:   FPGA bitstream\n");
    fprintf(stderr,"\n");
    fprintf(stderr, "if n > 0, points will be randomly generated instead of reading from infile.\n");
    exit(1);
  }
  int arg_count = 1;
  int kmin = atoi(argv[arg_count++]);
  int kmax = atoi(argv[arg_count++]);
  int dim = atoi(argv[arg_count++]);
  if(dim > DIM){
    printf("dimension should not exceed 256\n");
    exit(1);
  }
  int n = atoi(argv[arg_count++]);
  int clustersize = atoi(argv[arg_count++]);
  char *outfilename = new char[MAXNAMESIZE];
  char *infilename = new char[MAXNAMESIZE];
  strcpy(infilename, argv[arg_count++]);
  strcpy(outfilename, argv[arg_count++]);

  srand48(SEED);
  PStream* stream;
  if( n > 0 ) {
    stream = new SimStream(n);
  }
  else {
    stream = new FileStream(infilename);
  }

  _clInit(argv[arg_count++]);

  double t1 = gettime();
  streamCluster(stream, kmin, kmax, dim, clustersize, outfilename );
  double t2 = gettime();
  printf("time = %lf\n",t2-t1);

  _clClean();
  delete stream;
  
  return 0;
}
