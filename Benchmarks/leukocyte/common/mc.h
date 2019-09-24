#ifndef MC_H
#define MC_H



#ifndef LARGE_BUS
#define LARGE_BUS 512
#endif 


#ifndef SIZE_1
#define SIZE_1 1
#endif 


#ifndef SIZE_2
#define SIZE_2 1
#endif 


#ifndef SIZE_3
#define SIZE_3 1
#endif 


#ifndef DIM
#define DIM 1
#endif


#define MARS_WIDE_BUS_TYPE ap_uint<LARGE_BUS>


#include"ap_int.h"


#if (DIM == 2)
#include"./mars_wide_bus_2d.h"

#elif (DIM == 3)
#include"./mars_wide_bus_3d.h"

#elif (DIM == 4)
#include"./mars_wide_bus_4d.h"

#else
#include"./mars_wide_bus.h"

#endif



#endif
