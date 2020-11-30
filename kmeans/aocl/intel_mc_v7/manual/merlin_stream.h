/************************************************************************************
 *  (c) Copyright 2014-2015 Falcon Computing Solutions, Inc. All rights reserved.
 *
 *  This file contains confidential and proprietary information
 *  of Falcon Computing Solutions, Inc. and is protected under U.S. and
 *  international copyright and other intellectual property laws.
 *
 ************************************************************************************/

#ifndef __MERLIN_STREAM_H__
#define __MERLIN_STREAM_H__
struct merlin_stream {
  unsigned long buffer_size;
  unsigned long data_size;
  unsigned long curr_read_pos;
  unsigned long curr_write_pos;
  void *buffer;
};
typedef struct merlin_stream merlin_stream;

#ifdef __cplusplus
extern "C" {
#endif
void merlin_stream_init(merlin_stream *var, unsigned long buffer_size, 
                        unsigned long data_size);

void merlin_stream_reset(merlin_stream *var);

void merlin_stream_write(merlin_stream *var, void *data_in);

void merlin_stream_read(merlin_stream *var, void *data_out);
#ifdef __cplusplus
}
#endif
#endif
