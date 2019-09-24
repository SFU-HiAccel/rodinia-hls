/************************************************************************************
 *  (c) Copyright 2014-2015 Falcon Computing Solutions, Inc. All rights reserved.
 *
 *  This file contains confidential and proprietary information
 *  of Falcon Computing Solutions, Inc. and is protected under U.S. and
 *  international copyright and other intellectual property laws.
 *
 ************************************************************************************/

/************************************************************************************
 *  (c) Copyright 2014-2015 Falcon Computing Solutions, Inc. All rights reserved.
 *
 *  This file contains confidential and proprietary information
 *  of Falcon Computing Solutions, Inc. and is protected under U.S. and
 *  international copyright and other intellectual property laws.
 *
 ************************************************************************************/

#ifndef __MERLING_MEMCPY_512_INTERFACE_H_
#define __MERLING_MEMCPY_512_INTERFACE_H_
#ifdef __cplusplus
#include "ap_int.h"
#else
#include "ap_cint.h"
#endif
#define LARGE_BUS 512
#ifdef __cplusplus
//typedef ap_uint<LARGE_BUS> MARS_WIDE_BUS_TYPE;
#define MARS_WIDE_BUS_TYPE ap_uint<512>
#else
//typedef uint512 MARS_WIDE_BUS_TYPE;
#define MARS_WIDE_BUS_TYPE uint512
#endif
#include "mars_wide_bus.h"
#undef LARGE_BUS
#undef MARS_WIDE_BUS_TYPE
#endif
