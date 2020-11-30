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

#ifndef _XFCOMPRESSION_ZLIB_TREEGEN_MM_HPP_
#define _XFCOMPRESSION_ZLIB_TREEGEN_MM_HPP_

/**
 * @file treegen_kernel.hpp
 * @brief Header for tree generator kernel used in zlib compression.
 *
 * This file is part of Vitis Data Compression Library.
 */

//#include "zlib_config.hpp"
//
//#include <stdio.h>
#include <stdint.h>
//#include <assert.h>
#include "hls_stream.h"
#include <ap_int.h>
//#include "huffman_treegen.hpp"

//extern "C" {
/**
 * @brief This is an initial version of dynamic huffman codes & bit
 * length generation kernel which takes literal and distance frequency data as input
 * and generates dynamic huffman codes and bit length data. This version of the
 * kernel performs better for larger data sets in synchronization with LZ77
 * and Huffman Kernels. It need to be optimized further to achieve better
 * results for smaller block sizes (<1MB) so that it improves for smaller file
 * usecase.
 *
 * @param dyn_ltree_freq input literal frequency data
 * @param dyn_dtree_freq input distance frequency data
 * @param dyn_bltree_freq output bit-length frequency data
 * @param dyn_ltree_codes output literal codes
 * @param dyn_dtree_codes output distance codes
 * @param dyn_bltree_codes output bit-length codes
 * @param dyn_ltree_blen output literal bit length data
 * @param dyn_dtree_blen output distance bit length data
 * @param dyn_bltree_blen output bit-length of bitlengths data
 * @param max_codes output upper limit of codes for literal, distances,
 * bitlengths
 * @param block_size_in_kb input block size in bytes
 * @param input_size input data size
 * @param blocks_per_chunk number of blocks persent in current input data
 *
 */
void xilTreegenKernel(hls::stream<uint32_t> dyn_ltree_freq[PARALLEL_BLOCK],
                      hls::stream<uint32_t> dyn_dtree_freq[PARALLEL_BLOCK],
                      hls::stream<uint16_t> codeStream[PARALLEL_BLOCK],
                      hls::stream<uint8_t>  codeSize[PARALLEL_BLOCK],
                      
                      //hls::stream<ap_uint<32> > lz77_data_in[PARALLEL_BLOCK],
                      //hls::stream<bool> lz77_eos_in[PARALLEL_BLOCK],
                      //ap_uint<32> local_buffer[PARALLEL_BLOCK][65536],
                      //hls::stream<uint32_t> in_size[PARALLEL_BLOCK],
                      hls::stream<uint32_t>lz77_compressed_size_in[PARALLEL_BLOCK],
                      //hls::stream<ap_uint<32> > lz77_data_out[PARALLEL_BLOCK],
                      hls::stream<uint32_t> lz77_compressed_size_out[PARALLEL_BLOCK],
                      
                      uint32_t block_size_in_kb,
                      uint32_t input_size,
                      //uint32_t output_size[1],
                      uint32_t blocks_per_chunk);
//}

#endif // _XFCOMPRESSION_TREEGEN_KERNEL_HPP_
