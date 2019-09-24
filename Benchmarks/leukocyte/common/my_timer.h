#ifndef MY_TIMER_H
#define MY_TIMER_H

#include <sys/time.h>
#include <time.h>
#include <stdio.h>

typedef struct timespec timespec;
timespec diff(timespec start, timespec end)
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

timespec sum(timespec t1, timespec t2) {
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

void printTimeSpec(timespec t, const char* prefix) {
  printf("%s: %d.%09d\n", prefix, (int)t.tv_sec, (int)t.tv_nsec);
}

timespec tic( )
{
    timespec start_time;
    clock_gettime(CLOCK_REALTIME, &start_time);
    return start_time;
}

void toc( timespec* start_time, const char* prefix )
{
    timespec current_time;
    clock_gettime(CLOCK_REALTIME, &current_time);
    printTimeSpec( diff( *start_time, current_time ), prefix );
    *start_time = current_time;
}

#endif
