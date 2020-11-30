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
/**
 * @file xil_huffman_kernel.cpp
 * @brief Source for huffman kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */
//#include "zlib_huffman_enc_mm.hpp"
//#include "zlib_top.hpp"
//////////////
#include "lz_optional.hpp"
#include "stream_downsizer.hpp"
#include "mm2s.hpp"
#include "s2mm.hpp"
#include "stream_upsizer.hpp"

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include "hls_stream.h"
#include <ap_int.h>
#include "zlib_config.hpp"
#include "huffman_encoder.hpp"
#define MIN_BLOCK_SIZE 116

#define GMEM_DWIDTH 512
#define GMEM_BURST_SIZE 16

// DYNAMIC HUFFMAN Compress STATES
#define WRITE_TOKEN 0
#define ML_DIST_REP 1
#define LIT_REP 2
#define SEND_OUTPUT 3
#define ML_EXTRA 4
#define DIST_REP 5
#define DIST_EXTRA 6

// LZ specific Defines
#define BIT 8

#define d_code(dist, dist_code) ((dist) < 256 ? dist_code[dist] : dist_code[256 + ((dist) >> 7)])
//////////////

// 64bits/8bit = 8 Bytes
typedef ap_uint<16> uintOutV_t;

// 4 Bytes containing LL (1), ML (1), OFset (2)
typedef ap_uint<32> encoded_dt;

// 8 * 4 = 32 Bytes containing LL (1), ML (1), OFset (2)
typedef ap_uint<32> encodedV_dt;

void huffmanCore(hls::stream<encodedV_dt>& inStream,
                 hls::stream<xf::compression::uintMemWidth_t>& outStream512,
                 hls::stream<bool>& outStream512Eos,
                 uint32_t input_size,
                 hls::stream<uint16_t>& inStreamTree,
                 hls::stream<uint8_t>&  inStreamSize,
                 hls::stream<uint32_t>& compressedSize) {
    const uint32_t c_gmemBSize = 32;

    hls::stream<uintOutV_t> huffOut("huffOut");
    hls::stream<bool> huffOutEos("huffOutEos");
    hls::stream<uint16_t> bitVals("bitVals");
    hls::stream<uint8_t> bitLen("bitLen");
#pragma HLS STREAM variable = huffOut depth = 2048
#pragma HLS STREAM variable = huffOutEos depth = c_gmemBSize
#pragma HLS STREAM variable = bitVals depth = c_gmemBSize
#pragma HLS STREAM variable = bitLen depth = c_gmemBSize

#pragma HLS RESOURCE variable = huffOut core = FIFO_SRL
#pragma HLS RESOURCE variable = huffOutEos core = FIFO_SRL
#pragma HLS RESOURCE variable = bitVals core = FIFO_SRL
#pragma HLS RESOURCE variable = bitLen core = FIFO_SRL

#pragma HLS dataflow
    xf::compression::huffmanEncoder(inStream, bitVals, bitLen, input_size, inStreamTree, inStreamSize);

    xf::compression::details::bitPacking(bitVals, bitLen, huffOut, huffOutEos, compressedSize);

    xf::compression::upsizerEos<16, GMEM_DWIDTH>(huffOut, huffOutEos, outStream512, outStream512Eos);
    //xf::compression::upsizerEos<uint16_t, 16, GMEM_DWIDTH>(huffOut, huffOutEos, outStream512, outStream512Eos);
}

void huffman(hls::stream<encodedV_dt> in[PARALLEL_BLOCK],
             //ap_uint<32> in[PARALLEL_BLOCK][65536],
             //uint32_t in_size[PARALLEL_BLOCK],
             xf::compression::uintMemWidth_t* out,
             uint32_t input_idx[PARALLEL_BLOCK],
             uint32_t output_idx[PARALLEL_BLOCK],
             uint32_t input_size[PARALLEL_BLOCK],
             uint32_t output_size[PARALLEL_BLOCK],
             hls::stream<uint16_t> inStreamTree[PARALLEL_BLOCK],
             hls::stream<uint8_t> inStreamSize[PARALLEL_BLOCK],
             uint32_t n_blocks) {
    const uint32_t c_gmemBSize = 1024;

    //hls::stream<xf::compression::uintMemWidth_t> inStream512[PARALLEL_BLOCK];
    hls::stream<bool> outStream512Eos[PARALLEL_BLOCK];
    hls::stream<xf::compression::uintMemWidth_t> outStream512[PARALLEL_BLOCK];

    //hls::stream<uint32_t> strlitmtree_codes[PARALLEL_BLOCK];
    //hls::stream<uint32_t> strlitmtree_blen[PARALLEL_BLOCK];
    //hls::stream<uint32_t> strdistree_codes[PARALLEL_BLOCK];
    //hls::stream<uint32_t> strdtree_blen[PARALLEL_BLOCK];
    //hls::stream<uint32_t> strbitlentree_codes[PARALLEL_BLOCK];
    //hls::stream<uint32_t> strbitlentree_blen[PARALLEL_BLOCK];
    //hls::stream<uint32_t> strmax_code[PARALLEL_BLOCK];
#pragma HLS STREAM variable = outStream512Eos   depth = 32
//#pragma HLS STREAM variable = inStream512       depth = 32
#pragma HLS STREAM variable = outStream512      depth = 32
//#pragma HLS STREAM variable = strlitmtree_codes     depth = c_gmemBSize
//#pragma HLS STREAM variable = strlitmtree_blen      depth = c_gmemBSize
//#pragma HLS STREAM variable = strdistree_codes      depth = c_gmemBSize
//#pragma HLS STREAM variable = strdtree_blen         depth = c_gmemBSize
//#pragma HLS STREAM variable = strbitlentree_codes   depth = c_gmemBSize
//#pragma HLS STREAM variable = strbitlentree_blen    depth = c_gmemBSize
//#pragma HLS STREAM variable = strmax_code           depth = c_gmemBSize

#pragma HLS RESOURCE variable = outStream512Eos     core = FIFO_SRL
//#pragma HLS RESOURCE variable = inStream512         core = FIFO_SRL
#pragma HLS RESOURCE variable = outStream512        core = FIFO_SRL

    hls::stream<uint32_t> compressedSize[PARALLEL_BLOCK];
#pragma HLS STREAM variable = compressedSize depth = c_gmemBSize
#pragma HLS RESOURCE variable = compressedSize core = FIFO_SRL
#pragma HLS dataflow
    for (int i = 0; i < PARALLEL_BLOCK; i++) {
    #pragma HLS unroll
        huffmanCore(in[i], outStream512[i], outStream512Eos[i], input_size[i], inStreamTree[i], inStreamSize[i], compressedSize[i]);
    }
    xf::compression::s2mmEosNb<uint32_t, GMEM_BURST_SIZE, GMEM_DWIDTH, PARALLEL_BLOCK>(
        out, output_idx, outStream512, outStream512Eos, compressedSize, output_size);
}

typedef ap_axiu<32,0,0,0> datap;
//extern "C" {
void xilHuffmanKernel(//uint32_t wr_block_idx,
                      hls::stream<encodedV_dt> in[PARALLEL_BLOCK],
                      hls::stream<datap> &meter_out,
                      //ap_uint<32> in[PARALLEL_BLOCK][65536],
                      //uint32_t in_size[PARALLEL_BLOCK],
                      xf::compression::uintMemWidth_t* out,
                      hls::stream<uint32_t> in_block_size[PARALLEL_BLOCK],
                      uint32_t compressd_size[256],
                      hls::stream<uint16_t> inStreamTree[PARALLEL_BLOCK],       
                      hls::stream<uint8_t> inStreamSize[PARALLEL_BLOCK],       
                      uint32_t block_size_in_kb,
                      uint32_t input_size) {

    int block_idx = 0;
    int block_length = block_size_in_kb * 1024;
    int no_blocks = (input_size - 1) / block_length + 1;
    uint32_t max_block_size = block_size_in_kb * 1024 *2;

    bool small_block[PARALLEL_BLOCK];
    uint32_t input_block_size[PARALLEL_BLOCK];
    uint32_t input_idx[PARALLEL_BLOCK];
    uint32_t output_idx[PARALLEL_BLOCK];
    uint32_t output_block_size[PARALLEL_BLOCK];
    uint32_t small_block_inSize[PARALLEL_BLOCK];
    uint32_t max_lit_limit[PARALLEL_BLOCK];
#pragma HLS ARRAY_PARTITION variable = input_block_size dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = input_idx dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = output_idx dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = output_block_size dim = 0 complete
#pragma HLS ARRAY_PARTITION variable = max_lit_limit dim = 0 complete
    // Figure out total blocks and block sizes
    datap temp_meter_data;
    temp_meter_data.data = 1;
    for (int i = 0; i < no_blocks; i += PARALLEL_BLOCK) {
        int n_blocks = PARALLEL_BLOCK;
        if ((i + PARALLEL_BLOCK) > no_blocks) n_blocks = no_blocks - i;
        printf("huffman start !!\n");
        for (int j = 0; j < PARALLEL_BLOCK; j++) {
        #pragma HLS unroll
                uint32_t inBlockSize = in_block_size[j].read();
                if (inBlockSize < MIN_BLOCK_SIZE) {
                    small_block[j] = 1;
                    small_block_inSize[j] = inBlockSize;
                    input_block_size[j] = 0;
                    input_idx[j] = 0;
                } else {
                    small_block[j] = 0;
                    input_block_size[j] = inBlockSize;
                    input_idx[j] = (i + j) * max_block_size * 4;
                    output_idx[j] = (i + j) * max_block_size;
                }
            output_block_size[j] = 0;
            max_lit_limit[j] = 0;
        }

        huffman(in, out, input_idx, output_idx, input_block_size, output_block_size, inStreamTree, inStreamSize, n_blocks);

        for (int k = 0; k < n_blocks; k++) {
            if (max_lit_limit[k]) {
                compressd_size[k + i] = input_block_size[k];
            } else {
                compressd_size[k + i] = output_block_size[k];
            }
            if (small_block[k] == 1) compressd_size[k + i] = small_block_inSize[k];
        }
    } // Main loop ends here

    for(int i = 0; i < input_size; i += 1024){
        meter_out.write(temp_meter_data);
        }
    
}
//}
