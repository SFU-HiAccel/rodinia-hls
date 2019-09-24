#ifndef _CL_HELPER_
#define _CL_HELPER_

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "constant.h"
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
int load_file_to_memory(const char *filename, char **result);
void _clInit(char* bitstream);

double _clTask(
	float* coord, 
	float* weight, 
	float* target, 
	float* cost, 
	int* assign, 
	int* center_table, 
	char* switch_membership, 
	float* work_mem, 
	int num, 
	int numcenter,
	timespec* timers
);


void _clClean();
#endif //_CL_HELPER_
