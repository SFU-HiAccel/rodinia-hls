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

#include <assert.h>
#include <stdlib.h>
#include "ap_int.h"
#define MARS_WIDE_BUS_TYPE ap_uint<512>

#define BUS_WIDTH 64
#define cpp_get_range(tmp, x, y) tmp(x, y)
#define c_get_range(tmp, x, y) apint_get_range(tmp, x, y)
#define cpp_set_range(tmp, x, y, val) tmp(x, y) = val
#define c_set_range(tmp, x, y, val) tmp = apint_set_range(tmp, x, y, val)
#ifdef __cplusplus
#define tmp2(x, y) cpp_get_range(tmp, x, y)
#define tmp3(x, y, val) cpp_set_range(tmp, x, y, val)
#else
#define tmp2(x, y) c_get_range(tmp, x, y)
#define tmp3(x, y, val) c_set_range(tmp, x, y, val)
#endif

static void memcpy_wide_bus_read_int(int *a_buf, MARS_WIDE_BUS_TYPE *a,
                                            size_t offset_byte,
                                            size_t size_byte) {
#pragma HLS inline self
  const size_t data_width = sizeof(int);
  const size_t bus_width = BUS_WIDTH;
  const size_t num_elements = bus_width / data_width;
  size_t buf_size = size_byte / data_width;
  size_t offset = offset_byte / data_width;
  size_t head_align = offset & (num_elements - 1);
  size_t new_offset = offset + buf_size;
  size_t tail_align = (new_offset - 1) & (num_elements - 1);
  size_t start = offset / num_elements;
  size_t end = (offset + buf_size + num_elements - 1) / num_elements;
  //MARS_WIDE_BUS_TYPE *a_offset = a + start;
  size_t i, j;
  int len = end - start;
  assert(len <= buf_size / num_elements + 2);
  assert(len >= buf_size / num_elements);
  if (1 == len) {
#ifdef __cplusplus
    MARS_WIDE_BUS_TYPE tmp(a[start]);
#else
    MARS_WIDE_BUS_TYPE tmp = a[start];
#endif
    for (j = 0; j < num_elements; ++j) {
       if (j < head_align || j > tail_align)
         continue;
        a_buf[j - head_align] =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
    }
    return;
  }

  for (i = 0; i < len; ++i) {
#pragma HLS pipeline
#ifdef __cplusplus
    MARS_WIDE_BUS_TYPE tmp(a[i + start]);
#else
    MARS_WIDE_BUS_TYPE tmp = a[i + start];
#endif
    if (head_align == 0) {
      for (j = 0; j < num_elements; ++j) {
        if (i == end - start - 1 && j > tail_align)
          continue;
        a_buf[i * num_elements + j - 0] =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
      }
    }

    else if (head_align == 1) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 1)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        a_buf[i * num_elements + j - 1] =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
      }
    }

    else if (head_align == 2) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 2)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        a_buf[i * num_elements + j - 2] =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
      }
    }

    else if (head_align == 3) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 3)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        a_buf[i * num_elements + j - 3] =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
      }
    }

    else if (head_align == 4) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 4)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        a_buf[i * num_elements + j - 4] =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
      }
    }

    else if (head_align == 5) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 5)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        a_buf[i * num_elements + j - 5] =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
      }
    }

    else if (head_align == 6) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 6)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        a_buf[i * num_elements + j - 6] =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
      }
    }

    else if (head_align == 7) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 7)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        a_buf[i * num_elements + j - 7] =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
      }
    }

    else if (head_align == 8) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 8)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        a_buf[i * num_elements + j - 8] =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
      }
    }

    else if (head_align == 9) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 9)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        a_buf[i * num_elements + j - 9] =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
      }
    }

    else if (head_align == 10) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 10)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        a_buf[i * num_elements + j - 10] =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
      }
    }

    else if (head_align == 11) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 11)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        a_buf[i * num_elements + j - 11] =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
      }
    }

    else if (head_align == 12) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 12)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        a_buf[i * num_elements + j - 12] =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
      }
    }

    else if (head_align == 13) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 13)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        a_buf[i * num_elements + j - 13] =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
      }
    }

    else if (head_align == 14) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 14)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        a_buf[i * num_elements + j - 14] =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
      }
    }

    else {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 15)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        a_buf[i * num_elements + j - 15] =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
      }
    }
  }
}

static void memcpy_wide_bus_write_char(MARS_WIDE_BUS_TYPE *c, char *c_buf,
                                             size_t offset_byte,
                                             size_t size_byte) {
#pragma HLS inline self 
  const size_t data_width = sizeof(char);
  const size_t bus_width = BUS_WIDTH;
  const size_t num_elements = bus_width / data_width;
  size_t buf_size = size_byte / data_width;
  size_t offset = offset_byte / data_width;
  size_t head_align = offset & (num_elements - 1);
  size_t new_offset = offset + buf_size;
  size_t tail_align = (new_offset - 1) & (num_elements - 1);
  size_t start = offset / num_elements;
  size_t end = (offset + buf_size + num_elements - 1) / num_elements;
  size_t len = end - start;
  size_t i, j;
  if (head_align == 0)
    len = (buf_size + num_elements - 1) / num_elements;
  size_t align = 0;
  if (len == 1) {
    MARS_WIDE_BUS_TYPE tmp;
    if (head_align != 0 || tail_align != (num_elements - 1))
      tmp = c[start];
    for (j = 0; j < num_elements; ++j) {
      if (j < head_align)
        continue;
      if (j > tail_align)
        continue;
      tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
           c_buf[j - head_align]);
    }
    c[start] = tmp;
    return;
  }
  if (head_align != 0) {
    MARS_WIDE_BUS_TYPE tmp = c[start];
    for (j = 0; j < num_elements; ++j) {
      if (j < head_align)
        continue;
      tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
           c_buf[j - head_align]);
    }
    c[start] = tmp;
    start++;
    align++;
  }
  if (tail_align != (num_elements - 1))
    align++;
  int burst_len = len - align;
  assert(burst_len <= buf_size / num_elements);
  for (i = 0; i < burst_len; ++i) {
#pragma HLS pipeline
    MARS_WIDE_BUS_TYPE tmp;
    if (head_align == 0) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j - 0]);
      }

    }
    else if (head_align == 1) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 63]);
      }
    }

    else if (head_align == 2) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 62]);
      }
    }

    else if (head_align == 3) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 61]);
      }
    }

    else if (head_align == 4) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 60]);
      }
    }

    else if (head_align == 5) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 59]);
      }
    }

    else if (head_align == 6) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 58]);
      }
    }

    else if (head_align == 7) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 57]);
      }
    }

    else if (head_align == 8) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 56]);
      }
    }

    else if (head_align == 9) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 55]);
      }
    }

    else if (head_align == 10) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 54]);
      }
    }

    else if (head_align == 11) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 53]);
      }
    }

    else if (head_align == 12) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 52]);
      }
    }

    else if (head_align == 13) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 51]);
      }
    }

    else if (head_align == 14) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 50]);
      }
    }

    else if (head_align == 15) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 49]);
      }
    }

    else if (head_align == 16) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 48]);
      }
    }

    else if (head_align == 17) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 47]);
      }
    }

    else if (head_align == 18) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 46]);
      }
    }

    else if (head_align == 19) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 45]);
      }
    }

    else if (head_align == 20) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 44]);
      }
    }

    else if (head_align == 21) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 43]);
      }
    }

    else if (head_align == 22) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 42]);
      }
    }

    else if (head_align == 23) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 41]);
      }
    }

    else if (head_align == 24) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 40]);
      }
    }

    else if (head_align == 25) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 39]);
      }
    }

    else if (head_align == 26) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 38]);
      }
    }

    else if (head_align == 27) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 37]);
      }
    }

    else if (head_align == 28) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 36]);
      }
    }

    else if (head_align == 29) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 35]);
      }
    }

    else if (head_align == 30) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 34]);
      }
    }

    else if (head_align == 31) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 33]);
      }
    }

    else if (head_align == 32) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 32]);
      }
    }

    else if (head_align == 33) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 31]);
      }
    }

    else if (head_align == 34) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 30]);
      }
    }

    else if (head_align == 35) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 29]);
      }
    }

    else if (head_align == 36) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 28]);
      }
    }

    else if (head_align == 37) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 27]);
      }
    }

    else if (head_align == 38) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 26]);
      }
    }

    else if (head_align == 39) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 25]);
      }
    }

    else if (head_align == 40) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 24]);
      }
    }

    else if (head_align == 41) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 23]);
      }
    }

    else if (head_align == 42) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 22]);
      }
    }

    else if (head_align == 43) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 21]);
      }
    }

    else if (head_align == 44) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 20]);
      }
    }

    else if (head_align == 45) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 19]);
      }
    }

    else if (head_align == 46) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 18]);
      }
    }

    else if (head_align == 47) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 17]);
      }
    }

    else if (head_align == 48) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 16]);
      }
    }

    else if (head_align == 49) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 15]);
      }
    }

    else if (head_align == 50) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 14]);
      }
    }

    else if (head_align == 51) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 13]);
      }
    }

    else if (head_align == 52) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 12]);
      }
    }

    else if (head_align == 53) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 11]);
      }
    }

    else if (head_align == 54) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 10]);
      }
    }

    else if (head_align == 55) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 9]);
      }
    }

    else if (head_align == 56) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 8]);
      }
    }

    else if (head_align == 57) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 7]);
      }
    }

    else if (head_align == 58) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 6]);
      }
    }

    else if (head_align == 59) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 5]);
      }
    }

    else if (head_align == 60) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 4]);
      }
    }

    else if (head_align == 61) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 3]);
      }
    }

    else if (head_align == 62) {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 2]);
      }
    }

    else {
      for (j = 0; j < num_elements; ++j) {
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
             c_buf[i * num_elements + j + 1]);
      }
    }

    c[i + start] = tmp;
  }
  if (tail_align != num_elements - 1) {
    MARS_WIDE_BUS_TYPE tmp = c[end - 1];
    size_t pos = (len - align) * num_elements;
    pos += (num_elements - head_align) % num_elements;
    for (j = 0; j < num_elements; ++j) {
      if (j > tail_align)
        continue;
      tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8),
           c_buf[pos + j]);
    }
    c[end - 1] = tmp;
  }
}

static void memcpy_wide_bus_read_float(float *a_buf,
                                              MARS_WIDE_BUS_TYPE *a,
                                              size_t offset_byte,
                                              size_t size_byte) {
#pragma HLS inline self
  const size_t data_width = sizeof(float);
  const size_t bus_width = BUS_WIDTH;
  const size_t num_elements = bus_width / data_width;
  size_t buf_size = size_byte / data_width;
  size_t offset = offset_byte / data_width;
  size_t head_align = offset & (num_elements - 1);
  size_t new_offset = offset + buf_size;
  size_t tail_align = (new_offset - 1) & (num_elements - 1);
  size_t start = offset / num_elements;
  size_t end = (offset + buf_size + num_elements - 1) / num_elements;
  //MARS_WIDE_BUS_TYPE *a_offset = a + start;
  size_t i, j;
  int len = end - start;
  assert(len <= buf_size / num_elements + 2);
  assert(len >= buf_size / num_elements);
  if (1 == len) {
#ifdef __cplusplus
    MARS_WIDE_BUS_TYPE tmp(a[start]);
#else
    MARS_WIDE_BUS_TYPE tmp = a[start];
#endif
    for (j = 0; j < num_elements; ++j) {
       if (j < head_align || j > tail_align)
         continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[j - head_align] = *(float *)(&raw_bits);
    }
    return;
  }

  for (i = 0; i < len; ++i) {
#pragma HLS pipeline
#ifdef __cplusplus
    MARS_WIDE_BUS_TYPE tmp(a[i + start]);
#else
    MARS_WIDE_BUS_TYPE tmp = a[i + start];
#endif

    if (head_align == 0) {
      for (j = 0; j < num_elements; ++j) {
        if (i == end - start - 1 && j > tail_align)
          continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[i * num_elements + j - 0] = *(float *)(&raw_bits);
      }
    }

    else if (head_align == 1) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 1)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[i * num_elements + j - 1] = *(float *)(&raw_bits);
      }
    }

    else if (head_align == 2) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 2)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[i * num_elements + j - 2] = *(float *)(&raw_bits);
      }
    }

    else if (head_align == 3) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 3)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[i * num_elements + j - 3] = *(float *)(&raw_bits);
      }
    }

    else if (head_align == 4) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 4)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[i * num_elements + j - 4] = *(float *)(&raw_bits);
      }
    }

    else if (head_align == 5) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 5)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[i * num_elements + j - 5] = *(float *)(&raw_bits);
      }
    }

    else if (head_align == 6) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 6)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[i * num_elements + j - 6] = *(float *)(&raw_bits);
      }
    }

    else if (head_align == 7) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 7)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[i * num_elements + j - 7] = *(float *)(&raw_bits);
      }
    }

    else if (head_align == 8) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 8)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[i * num_elements + j - 8] = *(float *)(&raw_bits);
      }
    }

    else if (head_align == 9) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 9)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[i * num_elements + j - 9] = *(float *)(&raw_bits);
      }
    }

    else if (head_align == 10) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 10)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[i * num_elements + j - 10] = *(float *)(&raw_bits);
      }
    }

    else if (head_align == 11) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 11)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[i * num_elements + j - 11] = *(float *)(&raw_bits);
      }
    }

    else if (head_align == 12) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 12)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[i * num_elements + j - 12] = *(float *)(&raw_bits);
      }
    }

    else if (head_align == 13) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 13)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[i * num_elements + j - 13] = *(float *)(&raw_bits);
      }
    }

    else if (head_align == 14) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 14)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[i * num_elements + j - 14] = *(float *)(&raw_bits);
      }
    }

    else {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 15)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[i * num_elements + j - 15] = *(float *)(&raw_bits);
      }
    }
  }
}


#undef MARS_WIDE_BUS_TYPE
//#undef BUS_WIDTH
//#undef cpp_get_range
//#undef c_get_range
//#undef cpp_set_range
//#undef c_set_range
//#undef tmp2
//#undef tmp3
