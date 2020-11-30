/*
 * (c) Copyright 2019 Xilinx, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef _XFCOMPRESSION_ZLIB_S2_PINGPONG_BUFFER_HPP_
#define _XFCOMPRESSION_ZLIB_S2_PINGPONG_BUFFER_HPP_
#include "hls_stream.h"
//
#include <stdio.h>
#include <stdint.h>
//#include <assert.h>
#include <ap_int.h>
void s2pp_buffer(    hls::stream<ap_uint<32> > instream[PARALLEL_BLOCK],
                     hls::stream<bool> instream_eos[PARALLEL_BLOCK],
                     hls::stream<ap_uint<32> > outstream[PARALLEL_BLOCK],
                     uint32_t noBlocks);

#endif // _XFCOMPRESSION_ZLIB_S2_PINGPONG_BUFFER_HPP_
