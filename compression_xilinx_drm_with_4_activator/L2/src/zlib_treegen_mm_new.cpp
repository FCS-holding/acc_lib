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
 * @file xil_treegen_kernel.cpp
 * @brief Source for treegen kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */
#include "zlib_treegen_mm.hpp"
//#include "zlib_top.hpp"
////////////
#include "zlib_config.hpp"

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
//#include "hls_stream.h"
#include <ap_int.h>
#include "huffman_treegen.hpp"
////////////

void treegen_simple_ch(
                      hls::stream<uint32_t>& dyn_ltree_freq,
                      hls::stream<uint32_t>& dyn_dtree_freq,
                      hls::stream<uint16_t>& codeStream,
                      hls::stream<uint8_t>& codeSize,
                      int core_idx
                      ) {
    #pragma HLS function_instantiate variable=core_idx
    // Core module for zlib huffman treegen
    // Using single array for Frequency and Codes for literals, distances and bit-lengths
    xf::compression::Frequency inFreq[c_litCodeCount + c_dstCodeCount + c_blnCodeCount + 2];
    // one value padding between codes of literals-distance-bitlengths, needed later
    xf::compression::Codeword outCodes[c_litCodeCount + c_dstCodeCount + c_blnCodeCount + 2];
    uint16_t maxCodes[3] = {0, 0, 0};

    uint16_t offsets[3] = {0, c_litCodeCount + 1, (c_litCodeCount + c_dstCodeCount + 2)};
    ap_uint<20> codePacked;

    // initialize all the memory
    for (int i = 0; i < (c_litCodeCount + c_dstCodeCount + c_blnCodeCount + 2); ++i) {
        outCodes[i].codeword = 0;
        outCodes[i].bitlength = 0;
    }

    int offset = 0;
    for (int i = 0; i < c_litCodeCount; ++i) {
#pragma HLS PIPELINE II = 1
        //inFreq[i] = (Frequency)lit_freq[i];
        inFreq[i] = (xf::compression::Frequency)dyn_ltree_freq.read();
        printf("lit freq data %d \n", (uint32_t)inFreq[i]);
    }
    offset = offsets[1]; // copy distances
    for (int i = 0; i < c_dstCodeCount; ++i) {
#pragma HLS PIPELINE II = 1
        //inFreq[i + offset] = (Frequency)dist_freq[i];
        inFreq[i + offset] = (xf::compression::Frequency)dyn_dtree_freq.read();
        printf("dis freq data %d \n",(uint32_t)inFreq[i + offset]);
    }
    offset = offsets[2];
    for (int i = 0; i < c_blnCodeCount; ++i) {
        inFreq[i + offset] = 0; // just initialize
    }

    // read freqStream and generate codes for it
    for (int i = 0; i < 3; ++i) {
        // construct the huffman tree and generate huffman codes
        xf::compression::details::huffConstructTree(&(inFreq[offsets[i]]), &(outCodes[offsets[i]]), &(maxCodes[i]), i);
        // only after codes have been generated for literals and distances
        if (i < 2) {
            // generate frequency data for bitlengths
            xf::compression::details::genBitLenFreq(&(outCodes[offsets[i]]), &(inFreq[offsets[2]]), maxCodes[i]);
        }
    }
     printf("lit max code %d \n", maxCodes[0]);
     printf("dst max code %d \n", maxCodes[1]);

    // get maxCodes count for bit-length codes
    // specific to huffman tree of bit-lengths
    uint8_t bitlen_vals[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
    uint16_t bl_mxc;
bltree_blen:
    for (bl_mxc = c_blnCodeCount - 1; bl_mxc >= 3; --bl_mxc) {
#pragma HLS PIPELINE II = 3
        if ((uint8_t)(outCodes[bitlen_vals[bl_mxc] + offsets[2]].bitlength) != 0) break;
    }
    maxCodes[2] = bl_mxc;
    printf("bl  max code %d \n", maxCodes[2]);

    // Code from Huffman Encoder
    //********************************************//
    // Start of block = 4 and len = 3
    uint32_t start_of_block = 0x30004;

    codeStream << 4;
    codeSize << 3;
    // lcodes
    codeStream << ((maxCodes[0] + 1) - 257);
    codeSize << 5;

    // dcodes
    codeStream << ((maxCodes[1] + 1) - 1);
    codeSize << 5;

    // blcodes
    codeStream << ((maxCodes[2] + 1) - 4);
    codeSize << 4;

    uint16_t bitIndex = offsets[2];
// Send BL length data
send_bltree:
    for (int rank = 0; rank < bl_mxc + 1; rank++) {
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 64
#pragma HLS PIPELINE II = 1
        codeStream << outCodes[bitIndex + bitlen_vals[rank]].bitlength;
        codeSize << 3;
    } // BL data copy loop

    // Send Bitlengths for Literal and Distance Tree
    for (int tree = 0; tree < 2; tree++) {
        uint8_t prevlen = 0; // Last emitted Length
        uint8_t curlen = 0;  // Length of Current Code
        uint8_t nextlen = (tree == 0) ? outCodes[0].bitlength : outCodes[offsets[1]].bitlength; // Length of next code
        uint8_t count = 0;
        int max_count = 7; // Max repeat count
        int min_count = 4; // Min repeat count

        if (nextlen == 0) {
            max_count = 138;
            min_count = 3;
        }

        uint16_t max_code = (tree == 0) ? maxCodes[0] : maxCodes[1];

        xf::compression::Codeword temp = outCodes[bitIndex + c_reusePrevBlen];
        uint16_t reuse_prev_code = temp.codeword;
        uint8_t reuse_prev_len = temp.bitlength;
        temp = outCodes[bitIndex + c_reuseZeroBlen];
        uint16_t reuse_zero_code = temp.codeword;
        uint8_t reuse_zero_len = temp.bitlength;
        temp = outCodes[bitIndex + c_reuseZeroBlen7];
        uint16_t reuse_zero7_code = temp.codeword;
        uint8_t reuse_zero7_len = temp.bitlength;

    send_ltree:
        for (int n = 0; n <= max_code; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 286 max = 286
            curlen = nextlen;
            nextlen =
                (tree == 0) ? outCodes[n + 1].bitlength : outCodes[offsets[1] + n + 1].bitlength; // Length of next code

            if (++count < max_count && curlen == nextlen) {
                continue;
            } else if (count < min_count) {
            lit_cnt:
                temp = outCodes[bitIndex + curlen];
                for (uint8_t cnt = count; cnt != 0; --cnt) {
#pragma HLS LOOP_TRIPCOUNT min = 10 max = 10
#pragma HLS PIPELINE II = 1
                    codeStream << temp.codeword;
                    codeSize << temp.bitlength;
                }
                count = 0;

            } else if (curlen != 0) {
                if (curlen != prevlen) {
                    temp = outCodes[bitIndex + curlen];
                    codeStream << temp.codeword;
                    codeSize << temp.bitlength;
                    count--;
                }
                codeStream << reuse_prev_code;
                codeSize << reuse_prev_len;

                codeStream << count - 3;
                codeSize << 2;

            } else if (count <= 10) {
                codeStream << reuse_zero_code;
                codeSize << reuse_zero_len;

                codeStream << count - 3;
                codeSize << 3;

            } else {
                codeStream << reuse_zero7_code;
                codeSize << reuse_zero7_len;

                codeStream << count - 11;
                codeSize << 7;
            }

            count = 0;
            prevlen = curlen;
            if (nextlen == 0) {
                max_count = 138, min_count = 3;
            } else if (curlen == nextlen) {
                max_count = 6, min_count = 3;
            } else {
                max_count = 7, min_count = 4;
            }
        }
    }
    codeSize << 0;
//********************************************//
send_ltrees:
    for (int i = 0; i < c_litCodeCount; ++i) {
#pragma HLS PIPELINE II = 1
        xf::compression::Codeword code = outCodes[i];
        // prepare packet as <<bitlen>..<code>>
        codeStream << code.codeword;
        codeSize << code.bitlength;
    }

send_dtrees:
    for (int i = 0; i < c_dstCodeCount; ++i) {
#pragma HLS PIPELINE II = 1
        xf::compression::Codeword code = outCodes[offsets[1] + i];
        // prepare packet as <<bitlen>..<code>>
        codeStream << code.codeword;
        codeSize << code.bitlength;
    }


}
void lz77_data_buffer_ch(
    uint32_t block_idx,
    hls::stream<ap_uint<32> >& lz77_in,
    hls::stream<bool>& lz77_in_eos,
    hls::stream<ap_uint<32> >& huffman_data_out
){
    //ap_uint<32> local_buffer[2][32768];
    ap_uint<64> local_buffer[2][16384];
    #pragma HLS array_partition variable=local_buffer dim=1 complete
    #pragma HLS RESOURCE variable = local_buffer core = XPM_MEMORY uram
    uint32_t data_size;
    uint32_t cnt = 0;
    uint32_t data_cnt = 0;
    bool done = false;
    uint8_t w_pipo_flag = block_idx % 2;
    uint8_t r_pipo_flag = block_idx % 2;
    uint8_t d_cnt = 0;
    ap_uint<64> tmp_data_in;
    for(;done == false;){
    #pragma HLS PIPELINE II = 1
        if(!lz77_in_eos.empty()){
            bool eos_flag = lz77_in_eos.read();
            uint32_t tmp = lz77_in.read();
            if(eos_flag){
                if(d_cnt == 1){
                    tmp_data_in.range(31,0) = 0;
                    local_buffer[w_pipo_flag][cnt] = tmp_data_in;
                    //data_cnt++;
                    cnt++;
                    }
                done = true;
                }
            else{
                if(d_cnt == 0){
                    tmp_data_in.range(31,0) = tmp;
                    tmp_data_in = tmp_data_in << 32;
                    d_cnt = d_cnt + 1;
                    data_cnt++;
                    }
                else{
                    tmp_data_in.range(31,0) = tmp;
                    local_buffer[w_pipo_flag][cnt] = tmp_data_in;
                    cnt++;
                    data_cnt++;
                    d_cnt = 0;
                    }
                }
            }
        }
    data_size = data_cnt;
    for(uint32_t i = 0; i < data_size; i++){
#pragma HLS pipeline II=1
        if(i%2 == 0){
            huffman_data_out << local_buffer[r_pipo_flag][i/2].range(63,32);  
            uint32_t tmp = local_buffer[r_pipo_flag][i/2].range(63,32);
            }
        else{
            huffman_data_out << local_buffer[r_pipo_flag][i/2].range(31,0);   
            uint32_t tmp = local_buffer[r_pipo_flag][i/2].range(31,0);
            }
        }
    }
void compressed_size_rd_wr(
    hls::stream<uint32_t> lz77_compressed_size_in[PARALLEL_BLOCK],
    hls::stream<uint32_t> lz77_compressed_size_out[PARALLEL_BLOCK]
    ){
        for(uint8_t i = 0; i < PARALLEL_BLOCK; i++){
        #pragma HLS unroll
            bool done = false;
            for(;done == false;){
            #pragma HLS pipeline II=1
                if(!lz77_compressed_size_in[i].empty()){
                    uint32_t tmp = lz77_compressed_size_in[i].read();
                    lz77_compressed_size_out[i] << tmp;
                    done = true;
                    }
                }
            }
    
    
    
    
    } 
//extern "C" {
void xilTreegenKernel(hls::stream<uint32_t> dyn_ltree_freq[PARALLEL_BLOCK],
                      hls::stream<uint32_t> dyn_dtree_freq[PARALLEL_BLOCK],
                      hls::stream<uint16_t> codeStream[PARALLEL_BLOCK],
                      hls::stream<uint8_t>  codeSize[PARALLEL_BLOCK],
                        
                      hls::stream<ap_uint<32> > lz77_data_in[PARALLEL_BLOCK],
                      hls::stream<bool> lz77_eos_in[PARALLEL_BLOCK],
                      hls::stream<uint32_t> lz77_compressed_size_in[PARALLEL_BLOCK],
                      hls::stream<ap_uint<32> > lz77_data_out[PARALLEL_BLOCK],
                      hls::stream<uint32_t> lz77_compressed_size_out[PARALLEL_BLOCK],

                      uint32_t block_size_in_kb,
                      uint32_t input_size,
                      uint32_t blocks_per_chunk) {
    
    const uint32_t c_gmemBSize = 1024;
    //output_size[0] = input_size;
    int block_length = block_size_in_kb * 1024;
    int no_blocks = (input_size - 1) / block_length + 1;
    uint32_t block_idx = 0;
    for(uint32_t b_idx = 0; b_idx < no_blocks; b_idx = b_idx + PARALLEL_BLOCK){
        printf("treegen start %d \n",b_idx);
        for (uint8_t core_idx = 0; core_idx < PARALLEL_BLOCK; core_idx++) {
        #pragma HLS UNROLL
        // Copy Frequencies of Literal, Match Length and Distances to local buffers
            treegen_simple_ch(
                dyn_ltree_freq[core_idx],
                dyn_dtree_freq[core_idx],
                codeStream[core_idx],
                codeSize[core_idx],
                core_idx
            );
        }
        #if PARALLEL_BLOCK==8
            lz77_data_buffer_ch(block_idx,lz77_data_in[0],lz77_eos_in[0],lz77_data_out[0]);
            lz77_data_buffer_ch(block_idx,lz77_data_in[1],lz77_eos_in[1],lz77_data_out[1]);
            lz77_data_buffer_ch(block_idx,lz77_data_in[2],lz77_eos_in[2],lz77_data_out[2]);
            lz77_data_buffer_ch(block_idx,lz77_data_in[3],lz77_eos_in[3],lz77_data_out[3]);
            lz77_data_buffer_ch(block_idx,lz77_data_in[4],lz77_eos_in[4],lz77_data_out[4]);
            lz77_data_buffer_ch(block_idx,lz77_data_in[5],lz77_eos_in[5],lz77_data_out[5]);
            lz77_data_buffer_ch(block_idx,lz77_data_in[6],lz77_eos_in[6],lz77_data_out[6]);
            lz77_data_buffer_ch(block_idx,lz77_data_in[7],lz77_eos_in[7],lz77_data_out[7]);
        #else
            lz77_data_buffer_ch(block_idx,lz77_data_in[0],lz77_eos_in[0],lz77_data_out[0]);
            lz77_data_buffer_ch(block_idx,lz77_data_in[1],lz77_eos_in[1],lz77_data_out[1]);
            lz77_data_buffer_ch(block_idx,lz77_data_in[2],lz77_eos_in[2],lz77_data_out[2]);
            lz77_data_buffer_ch(block_idx,lz77_data_in[3],lz77_eos_in[3],lz77_data_out[3]);
            lz77_data_buffer_ch(block_idx,lz77_data_in[4],lz77_eos_in[4],lz77_data_out[4]);
            lz77_data_buffer_ch(block_idx,lz77_data_in[5],lz77_eos_in[5],lz77_data_out[5]);
            lz77_data_buffer_ch(block_idx,lz77_data_in[6],lz77_eos_in[6],lz77_data_out[6]);
            lz77_data_buffer_ch(block_idx,lz77_data_in[7],lz77_eos_in[7],lz77_data_out[7]);
            lz77_data_buffer_ch(block_idx,lz77_data_in[8],lz77_eos_in[8],lz77_data_out[8]);
            lz77_data_buffer_ch(block_idx,lz77_data_in[9],lz77_eos_in[9],lz77_data_out[9]);
            lz77_data_buffer_ch(block_idx,lz77_data_in[10],lz77_eos_in[10],lz77_data_out[10]);
            lz77_data_buffer_ch(block_idx,lz77_data_in[11],lz77_eos_in[11],lz77_data_out[11]);

        #endif
            block_idx++;
            compressed_size_rd_wr(lz77_compressed_size_in,lz77_compressed_size_out);
    }
}
//}
