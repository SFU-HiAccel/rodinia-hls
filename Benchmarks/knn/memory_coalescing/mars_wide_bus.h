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



#ifdef __cplusplus

template <int SIZE>
inline static void memcpy_wide_bus_read_float_2d(float a_buf[][SIZE],
                                                 size_t index2_offset,
                                                 size_t index1_offset,
                                                 MARS_WIDE_BUS_TYPE *a,
                                                 size_t offset_byte,
                                                 size_t size_byte) {
#pragma HLS inline self
  //#pragma HLS array_partition variable = a_buf cyclic factor = 16
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
  size_t index2 = index2_offset, index1 = index1_offset;
  const size_t alignment = (SIZE % num_elements) == 0 &&
                            (index1_offset % SIZE) == 0;
  const size_t index_offset = SIZE * index2_offset + index1_offset;
  const size_t bound = SIZE / num_elements;
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
#pragma HLS unroll
       if (j < head_align || j > tail_align)
         continue;
       size_t buf_index = j - head_align + index_offset;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[buf_index / SIZE][buf_index % SIZE] = *(float *)(&raw_bits);
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
        if (alignment)
          a_buf[index2][index1 * num_elements + j] = *(float *)(&raw_bits);
        else {
          size_t buf_index = (i * num_elements + j - 0) + index_offset; 
          a_buf[buf_index / SIZE][ buf_index % SIZE] =
              *(float *)(&raw_bits);
        }
      }
      if (alignment) {
        index1++;
        if (index1 == bound) {
          index1 = 0;
          ++index2;
        }
      }
    }
#if 0
    else if (head_align == 1) {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < 1)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        a_buf[(i * num_elements + j - 1) / SIZE][(i * num_elements + j - 1) %
                                                 SIZE] = *(float *)(&raw_bits);
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
        a_buf[(i * num_elements + j - 2) / SIZE][(i * num_elements + j - 2) %
                                                 SIZE] = *(float *)(&raw_bits);
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
        a_buf[(i * num_elements + j - 3) / SIZE][(i * num_elements + j - 3) %
                                                 SIZE] = *(float *)(&raw_bits);
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
        a_buf[(i * num_elements + j - 4) / SIZE][(i * num_elements + j - 4) %
                                                 SIZE] = *(float *)(&raw_bits);
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
        a_buf[(i * num_elements + j - 5) / SIZE][(i * num_elements + j - 5) %
                                                 SIZE] = *(float *)(&raw_bits);
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
        a_buf[(i * num_elements + j - 6) / SIZE][(i * num_elements + j - 6) %
                                                 SIZE] = *(float *)(&raw_bits);
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
        a_buf[(i * num_elements + j - 7) / SIZE][(i * num_elements + j - 7) %
                                                 SIZE] = *(float *)(&raw_bits);
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
        a_buf[(i * num_elements + j - 8) / SIZE][(i * num_elements + j - 8) %
                                                 SIZE] = *(float *)(&raw_bits);
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
        a_buf[(i * num_elements + j - 9) / SIZE][(i * num_elements + j - 9) %
                                                 SIZE] = *(float *)(&raw_bits);
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
        a_buf[(i * num_elements + j - 10) / SIZE][(i * num_elements + j - 10) %
                                                  SIZE] = *(float *)(&raw_bits);
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
        a_buf[(i * num_elements + j - 11) / SIZE][(i * num_elements + j - 11) %
                                                  SIZE] = *(float *)(&raw_bits);
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
        a_buf[(i * num_elements + j - 12) / SIZE][(i * num_elements + j - 12) %
                                                  SIZE] = *(float *)(&raw_bits);
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
        a_buf[(i * num_elements + j - 13) / SIZE][(i * num_elements + j - 13) %
                                                  SIZE] = *(float *)(&raw_bits);
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
        a_buf[(i * num_elements + j - 14) / SIZE][(i * num_elements + j - 14) %
                                                  SIZE] = *(float *)(&raw_bits);
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
        a_buf[(i * num_elements + j - 15) / SIZE][(i * num_elements + j - 15) %
                                                  SIZE] = *(float *)(&raw_bits);
      }
    }
#else
    else {
      for (j = 0; j < num_elements; ++j) {
        if (i == 0 && j < head_align)
          continue;
        if (i == end - start - 1 && j > tail_align)
          continue;
        int raw_bits =
            tmp2(((j + 1) * data_width * 8 - 1), (j * data_width * 8));
        size_t buf_index = (i * num_elements + j - head_align) + index_offset; 
        a_buf[ buf_index / SIZE][ buf_index  %  SIZE] = *(float *)(&raw_bits);
      }
    }

#endif
  }
}

template <int SIZE>
inline static void memcpy_wide_bus_write_float_2d(MARS_WIDE_BUS_TYPE *c,
                                                  float c_buf[][SIZE],
                                                  size_t index2_offset,
                                                  size_t index1_offset,
                                                  size_t offset_byte,
                                                  size_t size_byte) {
#pragma HLS inline self
  //#pragma HLS array_partition variable = c_buf cyclic factor = 16
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
  size_t len = end - start;
  const size_t index_offset = index2_offset * SIZE + index1_offset;
  size_t i, j;
  if (head_align == 0)
    len = (buf_size + num_elements - 1) / num_elements;
  if (len == 1) {
    MARS_WIDE_BUS_TYPE tmp;
    if (head_align != 0 || tail_align != (num_elements - 1))
      tmp = c[start];
    for (j = 0; j < num_elements; ++j) {
#pragma HLS unroll
      if (j < head_align)
        continue;
      if (j > tail_align)
        continue;
      size_t buf_index = (j - head_align) + index_offset;
      float buf_tmp = c_buf[ buf_index / SIZE][ buf_index % SIZE];
      int raw_bits = *(int *)&buf_tmp;
      tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
    }
    c[start] = tmp;
    return;
  }
  unsigned align = 0;
  if (head_align != 0) {
    MARS_WIDE_BUS_TYPE tmp = c[start];
    for (j = 0; j < num_elements; ++j) {
#pragma HLS unroll
      if (j < head_align)
        continue;
      size_t buf_index = (j - head_align) + index_offset;
      float buf_tmp = c_buf[ buf_index / SIZE][ buf_index % SIZE];
      int raw_bits = *(int *)&buf_tmp;
      tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
    }
    c[start] = tmp;
    start++;
    align++;
  }
  if (tail_align != (num_elements - 1))
    align++;
  size_t index2 = index2_offset, index1 = index1_offset;
  const size_t alignment = (SIZE % num_elements) == 0 &&
                           (index1_offset % num_elements) == 0;
  const size_t bound = SIZE / num_elements;
  int burst_len = len - align;
  assert(burst_len <= buf_size / num_elements);
  for (i = 0; i < burst_len; ++i) {
#pragma HLS pipeline
    MARS_WIDE_BUS_TYPE tmp;
    if (head_align == 0) {
      for (j = 0; j < num_elements; ++j) {
        size_t buf_index = (i * num_elements + j - 0) + index_offset; 
        float buf_tmp = alignment
                            ? c_buf[index2][index1 * num_elements + j]
                            : c_buf[ buf_index / SIZE][ buf_index % SIZE];
        int raw_bits = *(int *)&buf_tmp;
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
      }
      if (alignment) {
        index1++;
        if (index1 == bound) {
          index1 = 0;
          ++index2;
        }
      }

    }
#if 0
    else if (head_align == 1) {
      for (j = 0; j < num_elements; ++j) {
        float buf_tmp = c_buf[(i * num_elements + j + 15) /
                              SIZE][(i * num_elements + j + 15) % SIZE];
        int raw_bits = *(int *)&buf_tmp;
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
      }
    }

    else if (head_align == 2) {
      for (j = 0; j < num_elements; ++j) {
        float buf_tmp = c_buf[(i * num_elements + j + 14) /
                              SIZE][(i * num_elements + j + 14) % SIZE];
        int raw_bits = *(int *)&buf_tmp;
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
      }
    }

    else if (head_align == 3) {
      for (j = 0; j < num_elements; ++j) {
        float buf_tmp = c_buf[(i * num_elements + j + 13) /
                              SIZE][(i * num_elements + j + 13) % SIZE];
        int raw_bits = *(int *)&buf_tmp;
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
      }
    }

    else if (head_align == 4) {
      for (j = 0; j < num_elements; ++j) {
        float buf_tmp = c_buf[(i * num_elements + j + 12) /
                              SIZE][(i * num_elements + j + 12) % SIZE];
        int raw_bits = *(int *)&buf_tmp;
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
      }
    }

    else if (head_align == 5) {
      for (j = 0; j < num_elements; ++j) {
        float buf_tmp = c_buf[(i * num_elements + j + 11) /
                              SIZE][(i * num_elements + j + 11) % SIZE];
        int raw_bits = *(int *)&buf_tmp;
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
      }
    }

    else if (head_align == 6) {
      for (j = 0; j < num_elements; ++j) {
        float buf_tmp = c_buf[(i * num_elements + j + 10) /
                              SIZE][(i * num_elements + j + 10) % SIZE];
        int raw_bits = *(int *)&buf_tmp;
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
      }
    }

    else if (head_align == 7) {
      for (j = 0; j < num_elements; ++j) {
        float buf_tmp = c_buf[(i * num_elements + j + 9) /
                              SIZE][(i * num_elements + j + 9) % SIZE];
        int raw_bits = *(int *)&buf_tmp;
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
      }
    }

    else if (head_align == 8) {
      for (j = 0; j < num_elements; ++j) {
        float buf_tmp = c_buf[(i * num_elements + j + 8) /
                              SIZE][(i * num_elements + j + 8) % SIZE];
        int raw_bits = *(int *)&buf_tmp;
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
      }
    }

    else if (head_align == 9) {
      for (j = 0; j < num_elements; ++j) {
        float buf_tmp = c_buf[(i * num_elements + j + 7) /
                              SIZE][(i * num_elements + j + 7) % SIZE];
        int raw_bits = *(int *)&buf_tmp;
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
      }
    }

    else if (head_align == 10) {
      for (j = 0; j < num_elements; ++j) {
        float buf_tmp = c_buf[(i * num_elements + j + 6) /
                              SIZE][(i * num_elements + j + 6) % SIZE];
        int raw_bits = *(int *)&buf_tmp;
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
      }
    }

    else if (head_align == 11) {
      for (j = 0; j < num_elements; ++j) {
        float buf_tmp = c_buf[(i * num_elements + j + 5) /
                              SIZE][(i * num_elements + j + 5) % SIZE];
        int raw_bits = *(int *)&buf_tmp;
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
      }
    }

    else if (head_align == 12) {
      for (j = 0; j < num_elements; ++j) {
        float buf_tmp = c_buf[(i * num_elements + j + 4) /
                              SIZE][(i * num_elements + j + 4) % SIZE];
        int raw_bits = *(int *)&buf_tmp;
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
      }
    }

    else if (head_align == 13) {
      for (j = 0; j < num_elements; ++j) {
        float buf_tmp = c_buf[(i * num_elements + j + 3) /
                              SIZE][(i * num_elements + j + 3) % SIZE];
        int raw_bits = *(int *)&buf_tmp;
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
      }
    }

    else if (head_align == 14) {
      for (j = 0; j < num_elements; ++j) {
        float buf_tmp = c_buf[(i * num_elements + j + 2) /
                              SIZE][(i * num_elements + j + 2) % SIZE];
        int raw_bits = *(int *)&buf_tmp;
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
      }
    }

    else {
      for (j = 0; j < num_elements; ++j) {
        float buf_tmp = c_buf[(i * num_elements + j + 1) /
                              SIZE][(i * num_elements + j + 1) % SIZE];
        int raw_bits = *(int *)&buf_tmp;
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
      }
    }
#else
    else {
      for (j = 0; j < num_elements; ++j) {
        size_t index = i * num_elements + j + num_elements - 
            head_align + index_offset;
        float buf_tmp = c_buf[index / SIZE][index % SIZE];
        int raw_bits = *(int *)&buf_tmp;
        tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
      }
    }

#endif
    c[i + start] = tmp;
  }
  if (tail_align != num_elements - 1) {
    MARS_WIDE_BUS_TYPE tmp = c[end - 1];
    size_t pos = (len - align) * num_elements + index_offset;
    pos += (num_elements - head_align) % num_elements;
    for (j = 0; j < num_elements; ++j) {
#pragma HLS unroll
      if (j > tail_align)
        continue;
      float buf_tmp = c_buf[(pos + j) / SIZE][(pos + j) % SIZE];
      int raw_bits = *(int *)&buf_tmp;
      tmp3(((j + 1) * data_width * 8 - 1), (j * data_width * 8), raw_bits);
    }
    c[end - 1] = tmp;
  }
}
#endif

#undef BUS_WIDTH
#undef cpp_get_range
#undef c_get_range
#undef cpp_set_range
#undef c_set_range
#undef tmp2
#undef tmp3
#undef MARS_WIDE_BUS_TYPE