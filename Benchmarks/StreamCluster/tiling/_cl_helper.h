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
	int numcenter
);


void _clClean();
#endif //_CL_HELPER_
