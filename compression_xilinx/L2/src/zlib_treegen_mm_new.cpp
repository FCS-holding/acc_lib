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
    }
    offset = offsets[1]; // copy distances
    for (int i = 0; i < c_dstCodeCount; ++i) {
#pragma HLS PIPELINE II = 1
        //inFreq[i + offset] = (Frequency)dist_freq[i];
        inFreq[i + offset] = (xf::compression::Frequency)dyn_dtree_freq.read();
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
//uint32_t lz77_data_s2m(
//    hls::stream<ap_uint<32> >& lz77_in,
//    hls::stream<bool>& lz77_in_eos,
//    ap_uint<64> local_buffer[32768]
//    //hls::stream<uint32_t>& output_size
//){
//        
//){
//    uint32_t data_size;
//    //data_size = input_size.read();
//    data_size = input_size;
//    printf("input size %d \n", data_size);
//    if(data_size == 0) return;
//    uint32_t addr_offset = 0;
//    for(uint32_t i = 0; i < data_size; i++){
//#pragma HLS pipeline II=1
//        if(i%2 == 0){
//            huffman_data_out << local_buffer[i/2].range(63,32);  
//            uint32_t tmp = local_buffer[i/2].range(63,32);
//            }
//        else{
//            huffman_data_out << local_buffer[i/2].range(31,0);   
//            uint32_t tmp = local_buffer[i/2].range(31,0);
//            }
//        }
//}
//void lz77_data_buffer_ch(
//    uint32_t no_blocks,
//    hls::stream<ap_uint<32> >& lz77_in,
//    hls::stream<bool>& lz77_in_eos,
//    hls::stream<ap_uint<32> >& huffman_data_out,
//    ap_uint<64> local_buffer_ping[16384],
//    ap_uint<64> local_buffer_pong[16384],
//    uint32_t ping_size[1],
//    uint32_t pong_size[1]
//){
//
//
//    if(no_blocks %2 == 0){
//        ping_size[0] =lz77_data_s2m(lz77_in, lz77_in_eos, local_buffer_ping);
//        lz77_data_m2s(pong_size[0], huffman_data_out, local_buffer_pong);
//        }
//    else{
//        pong_size[0] =lz77_data_s2m(lz77_in, lz77_in_eos, local_buffer_pong);
//        lz77_data_m2s(ping_size[0], huffman_data_out, local_buffer_ping);
//        }
////    ap_uint<64> local_buffer[32768];
////    #pragma HLS RESOURCE variable = local_buffer core = XPM_MEMORY uram
////    hls::stream<uint32_t> data_size;
////    #pragma HLS STREAM variable = data_size          depth = 16
////    for(int i = 0; i < no_blocks; i++){
//    //#pragma HLS dataflow
//        //lz77_data_s2m(i,lz77_in, lz77_in_eos, local_buffer, data_size);
//        //lz77_data_m2s(i,data_size, huffman_data_out, local_buffer);
////    }
////    uint32_t data_size;
////    uint32_t cnt = 0;
////    uint32_t data_cnt = 0;
////    bool done = false;
////    uint8_t w_pipo_flag = block_idx % 2;
////    uint8_t r_pipo_flag = block_idx % 2;
////    uint8_t d_cnt = 0;
////    ap_uint<64> tmp_data_in;
////    for(;done == false;){
////    #pragma HLS PIPELINE II = 1
////        if(!lz77_in_eos.empty()){
////            bool eos_flag = lz77_in_eos.read();
////            uint32_t tmp = lz77_in.read();
////            if(eos_flag){
////                if(d_cnt == 1){
////                    tmp_data_in.range(31,0) = 0;
////                    local_buffer[w_pipo_flag][cnt] = tmp_data_in;
////                    //data_cnt++;
////                    cnt++;
////                    }
////                done = true;
////                }
////            else{
////                if(d_cnt == 0){
////                    tmp_data_in.range(31,0) = tmp;
////                    tmp_data_in = tmp_data_in << 32;
////                    d_cnt = d_cnt + 1;
////                    data_cnt++;
////                    }
////                else{
////                    tmp_data_in.range(31,0) = tmp;
////                    local_buffer[w_pipo_flag][cnt] = tmp_data_in;
////                    cnt++;
////                    data_cnt++;
////                    d_cnt = 0;
////                    }
////                }
////            }
////        }
////    //output_size << data_cnt;
////    printf("data_cnt %d \n",data_cnt);
////    return data_cnt;
////}
////void lz77_data_m2s(
////    uint32_t input_size,
////    hls::stream<ap_uint<32> >&huffman_data_out,
////    ap_uint<64> local_buffer[32768]
////){
////    uint32_t data_size;
////    //data_size = input_size.read();
////    data_size = input_size;
////    printf("input size %d \n", data_size);
////    if(data_size == 0) return;
////    uint32_t addr_offset = 0;
////    for(uint32_t i = 0; i < data_size; i++){
////#pragma HLS pipeline II=1
////        if(i%2 == 0){
////            huffman_data_out << local_buffer[i/2].range(63,32);  
////            uint32_t tmp = local_buffer[i/2].range(63,32);
////            }
////        else{
////            huffman_data_out << local_buffer[i/2].range(31,0);   
////            uint32_t tmp = local_buffer[i/2].range(31,0);
////            }
////        }
////}
//void lz77_data_buffer_ch(
//    uint32_t no_blocks,
//    hls::stream<ap_uint<32> >& lz77_in,
//    hls::stream<bool>& lz77_in_eos,
//    hls::stream<ap_uint<32> >& huffman_data_out,
//    ap_uint<64> local_buffer_ping[16384],
//    ap_uint<64> local_buffer_pong[16384],
//    uint32_t ping_size[1],
//    uint32_t pong_size[1]
//){
//
//
//    if(no_blocks %2 == 0){
//        ping_size[0] =lz77_data_s2m(lz77_in, lz77_in_eos, local_buffer_ping);
//        lz77_data_m2s(pong_size[0], huffman_data_out, local_buffer_pong);
//        }
//    else{
//        pong_size[0] =lz77_data_s2m(lz77_in, lz77_in_eos, local_buffer_pong);
//        lz77_data_m2s(ping_size[0], huffman_data_out, local_buffer_ping);
//        }
////    ap_uint<64> local_buffer[32768];
////    #pragma HLS RESOURCE variable = local_buffer core = XPM_MEMORY uram
////    hls::stream<uint32_t> data_size;
////    #pragma HLS STREAM variable = data_size          depth = 16
////    for(int i = 0; i < no_blocks; i++){
//    //#pragma HLS dataflow
//        //lz77_data_s2m(i,lz77_in, lz77_in_eos, local_buffer, data_size);
//        //lz77_data_m2s(i,data_size, huffman_data_out, local_buffer);
////    }
////    uint32_t data_size;
////    uint32_t cnt = 0;
////    uint32_t data_cnt = 0;
////    bool done = false;
////    uint8_t w_pipo_flag = block_idx % 2;
////    uint8_t r_pipo_flag = block_idx % 2;
////    uint8_t d_cnt = 0;
////    ap_uint<64> tmp_data_in;
////    for(;done == false;){
////    #pragma HLS PIPELINE II = 1
////        if(!lz77_in_eos.empty()){
////            bool eos_flag = lz77_in_eos.read();
////            uint32_t tmp = lz77_in.read();
////            if(eos_flag){
////                if(d_cnt == 1){
////                    tmp_data_in.range(31,0) = 0;
////                    local_buffer[w_pipo_flag][cnt] = tmp_data_in;
////                    //data_cnt++;
////                    cnt++;
////                    }
////                done = true;
////                }
////            else{
////                if(d_cnt == 0){
////                    tmp_data_in.range(31,0) = tmp;
////                    tmp_data_in = tmp_data_in << 32;
////                    d_cnt = d_cnt + 1;
////                    data_cnt++;
////                    }
////                else{
////                    tmp_data_in.range(31,0) = tmp;
////                    local_buffer[w_pipo_flag][cnt] = tmp_data_in;
////                    cnt++;
////                    data_cnt++;
////                    d_cnt = 0;
////                    }
////                }
////            }
////        }
////    data_size = data_cnt;
////    for(uint32_t i = 0; i < data_size; i++){
////#pragma HLS pipeline II=1
////        if(i%2 == 0){
////            huffman_data_out << local_buffer[r_pipo_flag][i/2].range(63,32);  
////            uint32_t tmp = local_buffer[r_pipo_flag][i/2].range(63,32);
////            }
////        else{
////            huffman_data_out << local_buffer[r_pipo_flag][i/2].range(31,0);   
////            uint32_t tmp = local_buffer[r_pipo_flag][i/2].range(31,0);
////            }
////        }
//    }
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
//            lz77_data_buffer_ch(no_blocks,lz77_data_in[6],lz77_eos_in[6],lz77_data_out[6]);
//            lz77_data_buffer_ch(no_blocks,lz77_data_in[7],lz77_eos_in[7],lz77_data_out[7]);
//            lz77_data_buffer_ch(no_blocks,lz77_data_in[8],lz77_eos_in[8],lz77_data_out[8]);
//            lz77_data_buffer_ch(no_blocks,lz77_data_in[9],lz77_eos_in[9],lz77_data_out[9]);
//            lz77_data_buffer_ch(no_blocks,lz77_data_in[10],lz77_eos_in[10],lz77_data_out[10]);
//            lz77_data_buffer_ch(no_blocks,lz77_data_in[11],lz77_eos_in[11],lz77_data_out[11]);
//   
//    }
//extern "C" {
void xilTreegenKernel(hls::stream<uint32_t> dyn_ltree_freq[PARALLEL_BLOCK],
                      hls::stream<uint32_t> dyn_dtree_freq[PARALLEL_BLOCK],
                      hls::stream<uint16_t> codeStream[PARALLEL_BLOCK],
                      hls::stream<uint8_t>  codeSize[PARALLEL_BLOCK],
                        
                      //hls::stream<ap_uint<32> > lz77_data_in[PARALLEL_BLOCK],
                      //hls::stream<bool> lz77_eos_in[PARALLEL_BLOCK],
                      hls::stream<uint32_t> lz77_compressed_size_in[PARALLEL_BLOCK],
                      //hls::stream<ap_uint<32> > lz77_data_out[PARALLEL_BLOCK],
                      hls::stream<uint32_t> lz77_compressed_size_out[PARALLEL_BLOCK],

                      uint32_t block_size_in_kb,
                      uint32_t input_size,
                      uint32_t blocks_per_chunk) {
    
    const uint32_t c_gmemBSize = 1024;
    //output_size[0] = input_size;
    int block_length = block_size_in_kb * 1024;
    int no_blocks = (input_size - 1) / block_length + 1;
    uint32_t block_idx = 0;
    printf("treegen  start ! \n");
//    uint32_t ping_size[PARALLEL_BLOCK];
//#pragma HLS array_partition variable=ping_size dim=1 complete
//    uint32_t pong_size[PARALLEL_BLOCK];
//#pragma HLS array_partition variable=pong_size dim=1 complete
//    for(int j = 0; j < PARALLEL_BLOCK; j++){
//    #pragma HLS unroll
//        ping_size[j] = 0;
//        pong_size[j] = 0;
//        }
//    ap_uint<64> local_buffer_ping_0[16384];
//    #pragma HLS RESOURCE variable = local_buffer_ping_0 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_pong_0[16384];
//    #pragma HLS RESOURCE variable = local_buffer_pong_0 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_ping_1[16384];
//    #pragma HLS RESOURCE variable = local_buffer_ping_1 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_pong_1[16384];
//    #pragma HLS RESOURCE variable = local_buffer_pong_1 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_ping_2[16384];
//    #pragma HLS RESOURCE variable = local_buffer_ping_2 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_pong_2[16384];
//    #pragma HLS RESOURCE variable = local_buffer_pong_2 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_ping_3[16384];
//    #pragma HLS RESOURCE variable = local_buffer_ping_3 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_pong_3[16384];
//    #pragma HLS RESOURCE variable = local_buffer_pong_3 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_ping_4[16384];
//    #pragma HLS RESOURCE variable = local_buffer_ping_4 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_pong_4[16384];
//    #pragma HLS RESOURCE variable = local_buffer_pong_4 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_ping_5[16384];
//    #pragma HLS RESOURCE variable = local_buffer_ping_5 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_pong_5[16384];
//    #pragma HLS RESOURCE variable = local_buffer_pong_5 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_ping_6[16384];
//    #pragma HLS RESOURCE variable = local_buffer_ping_6 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_pong_6[16384];
//    #pragma HLS RESOURCE variable = local_buffer_pong_6 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_ping_7[16384];
//    #pragma HLS RESOURCE variable = local_buffer_ping_7 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_pong_7[16384];
//    #pragma HLS RESOURCE variable = local_buffer_pong_7 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_ping_8[16384];
//    #pragma HLS RESOURCE variable = local_buffer_ping_8 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_pong_8[16384];
//    #pragma HLS RESOURCE variable = local_buffer_pong_8 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_ping_9[16384];
//    #pragma HLS RESOURCE variable = local_buffer_ping_9 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_pong_9[16384];
//    #pragma HLS RESOURCE variable = local_buffer_pong_9 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_ping_10[16384];
//    #pragma HLS RESOURCE variable = local_buffer_ping_10 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_pong_10[16384];
//    #pragma HLS RESOURCE variable = local_buffer_pong_10 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_ping_11[16384];
//    #pragma HLS RESOURCE variable = local_buffer_ping_11 core = XPM_MEMORY uram
//    ap_uint<64> local_buffer_pong_11[16384];
//    #pragma HLS RESOURCE variable = local_buffer_pong_11 core = XPM_MEMORY uram
//    hls::stream<uint32_t> data_size[PARALLEL_BLOCK];
//    #pragma HLS STREAM variable = data_size          depth = 16
    //for(uint32_t t_idx = 0; t_idx < no_blocks; t_idx = t_idx + PARALLEL_BLOCK){
    //    if(block_idx % 2 == 0){
    //        printf("wr ping rd pong \n");
    //        ping_size[0] =lz77_data_s2m(lz77_data_in[0], lz77_eos_in[0], local_buffer_ping_0);
    //        ping_size[1] =lz77_data_s2m(lz77_data_in[1], lz77_eos_in[1], local_buffer_ping_1);
    //        ping_size[2] =lz77_data_s2m(lz77_data_in[2], lz77_eos_in[2], local_buffer_ping_2);
    //        ping_size[3] =lz77_data_s2m(lz77_data_in[3], lz77_eos_in[3], local_buffer_ping_3);
    //        ping_size[4] =lz77_data_s2m(lz77_data_in[4], lz77_eos_in[4], local_buffer_ping_4);
    //        ping_size[5] =lz77_data_s2m(lz77_data_in[5], lz77_eos_in[5], local_buffer_ping_5);
    //        ping_size[6] =lz77_data_s2m(lz77_data_in[6], lz77_eos_in[6], local_buffer_ping_6);
    //        ping_size[7] =lz77_data_s2m(lz77_data_in[7], lz77_eos_in[7], local_buffer_ping_7);
    //        ping_size[8] =lz77_data_s2m(lz77_data_in[8], lz77_eos_in[8], local_buffer_ping_8);
    //        ping_size[9] =lz77_data_s2m(lz77_data_in[9], lz77_eos_in[9], local_buffer_ping_9);
    //        ping_size[10] =lz77_data_s2m(lz77_data_in[10], lz77_eos_in[10], local_buffer_ping_10);
    //        ping_size[11] =lz77_data_s2m(lz77_data_in[11], lz77_eos_in[11], local_buffer_ping_11);
    //        lz77_data_m2s(pong_size[0], lz77_data_out[0], local_buffer_pong_0);
    //        lz77_data_m2s(pong_size[1], lz77_data_out[1], local_buffer_pong_1);
    //        lz77_data_m2s(pong_size[2], lz77_data_out[2], local_buffer_pong_2);
    //        lz77_data_m2s(pong_size[3], lz77_data_out[3], local_buffer_pong_3);
    //        lz77_data_m2s(pong_size[4], lz77_data_out[4], local_buffer_pong_4);
    //        lz77_data_m2s(pong_size[5], lz77_data_out[5], local_buffer_pong_5);
    //        lz77_data_m2s(pong_size[6], lz77_data_out[6], local_buffer_pong_6);
    //        lz77_data_m2s(pong_size[7], lz77_data_out[7], local_buffer_pong_7);
    //        lz77_data_m2s(pong_size[8], lz77_data_out[8], local_buffer_pong_8);
    //        lz77_data_m2s(pong_size[9], lz77_data_out[9], local_buffer_pong_9);
    //        lz77_data_m2s(pong_size[10], lz77_data_out[10], local_buffer_pong_10);
    //        lz77_data_m2s(pong_size[11], lz77_data_out[11], local_buffer_pong_11);
    //    }
    //    else{
    //        printf("wr ping rd pong \n");
    //        pong_size[0] =lz77_data_s2m(lz77_data_in[0], lz77_eos_in[0], local_buffer_pong_0);
    //        pong_size[1] =lz77_data_s2m(lz77_data_in[1], lz77_eos_in[1], local_buffer_pong_1);
    //        pong_size[2] =lz77_data_s2m(lz77_data_in[2], lz77_eos_in[2], local_buffer_pong_2);
    //        pong_size[3] =lz77_data_s2m(lz77_data_in[3], lz77_eos_in[3], local_buffer_pong_3);
    //        pong_size[4] =lz77_data_s2m(lz77_data_in[4], lz77_eos_in[4], local_buffer_pong_4);
    //        pong_size[5] =lz77_data_s2m(lz77_data_in[5], lz77_eos_in[5], local_buffer_pong_5);
    //        pong_size[6] =lz77_data_s2m(lz77_data_in[6], lz77_eos_in[6], local_buffer_pong_6);
    //        pong_size[7] =lz77_data_s2m(lz77_data_in[7], lz77_eos_in[7], local_buffer_pong_7);
    //        pong_size[8] =lz77_data_s2m(lz77_data_in[8], lz77_eos_in[8], local_buffer_pong_8);
    //        pong_size[9] =lz77_data_s2m(lz77_data_in[9], lz77_eos_in[9], local_buffer_pong_9);
    //        pong_size[10] =lz77_data_s2m(lz77_data_in[10], lz77_eos_in[10], local_buffer_pong_10);
    //        pong_size[11] =lz77_data_s2m(lz77_data_in[11], lz77_eos_in[11], local_buffer_pong_11);
    //        lz77_data_m2s(ping_size[0], lz77_data_out[0], local_buffer_ping_0);
    //        lz77_data_m2s(ping_size[1], lz77_data_out[1], local_buffer_ping_1);
    //        lz77_data_m2s(ping_size[2], lz77_data_out[2], local_buffer_ping_2);
    //        lz77_data_m2s(ping_size[3], lz77_data_out[3], local_buffer_ping_3);
    //        lz77_data_m2s(ping_size[4], lz77_data_out[4], local_buffer_ping_4);
    //        lz77_data_m2s(ping_size[5], lz77_data_out[5], local_buffer_ping_5);
    //        lz77_data_m2s(ping_size[6], lz77_data_out[6], local_buffer_ping_6);
    //        lz77_data_m2s(ping_size[7], lz77_data_out[7], local_buffer_ping_7);
    //        lz77_data_m2s(ping_size[8], lz77_data_out[8], local_buffer_ping_8);
    //        lz77_data_m2s(ping_size[9], lz77_data_out[9], local_buffer_ping_9);
    //        lz77_data_m2s(ping_size[10], lz77_data_out[10], local_buffer_ping_10);
    //        lz77_data_m2s(ping_size[11], lz77_data_out[11], local_buffer_ping_11);
    //        
    //    }
    //    block_idx++;
    //}
    for(uint32_t b_idx = 0; b_idx < no_blocks; b_idx = b_idx + PARALLEL_BLOCK){
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
        //#if PARALLEL_BLOCK==8
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[0],lz77_eos_in[0],lz77_data_out[0],local_buffer_ping_0,local_buffer_pong_0,&ping_size[0],&pong_size[0]);
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[1],lz77_eos_in[1],lz77_data_out[1],local_buffer_ping_1,local_buffer_pong_1,&ping_size[1],&pong_size[1]);
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[2],lz77_eos_in[2],lz77_data_out[2],local_buffer_ping_2,local_buffer_pong_2,&ping_size[2],&pong_size[2]);
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[3],lz77_eos_in[3],lz77_data_out[3],local_buffer_ping_3,local_buffer_pong_3,&ping_size[3],&pong_size[3]);
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[4],lz77_eos_in[4],lz77_data_out[4],local_buffer_ping_4,local_buffer_pong_4,&ping_size[4],&pong_size[4]);
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[5],lz77_eos_in[5],lz77_data_out[5],local_buffer_ping_5,local_buffer_pong_5,&ping_size[5],&pong_size[5]);
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[6],lz77_eos_in[6],lz77_data_out[6],local_buffer_ping_6,local_buffer_pong_6,&ping_size[6],&pong_size[6]);
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[7],lz77_eos_in[7],lz77_data_out[7],local_buffer_ping_7,local_buffer_pong_7,&ping_size[7],&pong_size[7]);
        //#else
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[0],lz77_eos_in[0],lz77_data_out[0],local_buffer_ping_0,local_buffer_pong_0,&ping_size[0],&pong_size[0]);
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[1],lz77_eos_in[1],lz77_data_out[1],local_buffer_ping_1,local_buffer_pong_1,&ping_size[1],&pong_size[1]);
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[2],lz77_eos_in[2],lz77_data_out[2],local_buffer_ping_2,local_buffer_pong_2,&ping_size[2],&pong_size[2]);
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[3],lz77_eos_in[3],lz77_data_out[3],local_buffer_ping_3,local_buffer_pong_3,&ping_size[3],&pong_size[3]);
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[4],lz77_eos_in[4],lz77_data_out[4],local_buffer_ping_4,local_buffer_pong_4,&ping_size[4],&pong_size[4]);
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[5],lz77_eos_in[5],lz77_data_out[5],local_buffer_ping_5,local_buffer_pong_5,&ping_size[5],&pong_size[5]);
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[6],lz77_eos_in[6],lz77_data_out[6],local_buffer_ping_6,local_buffer_pong_6,&ping_size[6],&pong_size[6]);
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[7],lz77_eos_in[7],lz77_data_out[7],local_buffer_ping_7,local_buffer_pong_7,&ping_size[7],&pong_size[7]);
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[8],lz77_eos_in[8],lz77_data_out[8],local_buffer_ping_8,local_buffer_pong_8,&ping_size[8],&pong_size[8]);
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[9],lz77_eos_in[9],lz77_data_out[9],local_buffer_ping_9,local_buffer_pong_9,&ping_size[9],&pong_size[9]);
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[10],lz77_eos_in[10],lz77_data_out[10],local_buffer_ping_10,local_buffer_pong_10,&ping_size[10],&pong_size[10]);
        //    lz77_data_buffer_ch(block_idx,lz77_data_in[11],lz77_eos_in[11],lz77_data_out[11],local_buffer_ping_11,local_buffer_pong_11,&ping_size[11],&pong_size[11]);
        /*
        if(block_idx % 2 == 0){
            printf("wr ping rd pong \n");
            ping_size[0] =lz77_data_s2m(lz77_data_in[0], lz77_eos_in[0], local_buffer_ping_0);
            ping_size[1] =lz77_data_s2m(lz77_data_in[1], lz77_eos_in[1], local_buffer_ping_1);
            ping_size[2] =lz77_data_s2m(lz77_data_in[2], lz77_eos_in[2], local_buffer_ping_2);
            ping_size[3] =lz77_data_s2m(lz77_data_in[3], lz77_eos_in[3], local_buffer_ping_3);
            ping_size[4] =lz77_data_s2m(lz77_data_in[4], lz77_eos_in[4], local_buffer_ping_4);
            ping_size[5] =lz77_data_s2m(lz77_data_in[5], lz77_eos_in[5], local_buffer_ping_5);
            ping_size[6] =lz77_data_s2m(lz77_data_in[6], lz77_eos_in[6], local_buffer_ping_6);
            ping_size[7] =lz77_data_s2m(lz77_data_in[7], lz77_eos_in[7], local_buffer_ping_7);
            ping_size[8] =lz77_data_s2m(lz77_data_in[8], lz77_eos_in[8], local_buffer_ping_8);
            ping_size[9] =lz77_data_s2m(lz77_data_in[9], lz77_eos_in[9], local_buffer_ping_9);
            ping_size[10] =lz77_data_s2m(lz77_data_in[10], lz77_eos_in[10], local_buffer_ping_10);
            ping_size[11] =lz77_data_s2m(lz77_data_in[11], lz77_eos_in[11], local_buffer_ping_11);
            lz77_data_m2s(pong_size[0], lz77_data_out[0], local_buffer_pong_0);
            lz77_data_m2s(pong_size[1], lz77_data_out[1], local_buffer_pong_1);
            lz77_data_m2s(pong_size[2], lz77_data_out[2], local_buffer_pong_2);
            lz77_data_m2s(pong_size[3], lz77_data_out[3], local_buffer_pong_3);
            lz77_data_m2s(pong_size[4], lz77_data_out[4], local_buffer_pong_4);
            lz77_data_m2s(pong_size[5], lz77_data_out[5], local_buffer_pong_5);
            lz77_data_m2s(pong_size[6], lz77_data_out[6], local_buffer_pong_6);
            lz77_data_m2s(pong_size[7], lz77_data_out[7], local_buffer_pong_7);
            lz77_data_m2s(pong_size[8], lz77_data_out[8], local_buffer_pong_8);
            lz77_data_m2s(pong_size[9], lz77_data_out[9], local_buffer_pong_9);
            lz77_data_m2s(pong_size[10], lz77_data_out[10], local_buffer_pong_10);
            lz77_data_m2s(pong_size[11], lz77_data_out[11], local_buffer_pong_11);
        }
        else{
            printf("wr ping rd pong \n");
            pong_size[0] =lz77_data_s2m(lz77_data_in[0], lz77_eos_in[0], local_buffer_pong_0);
            pong_size[1] =lz77_data_s2m(lz77_data_in[1], lz77_eos_in[1], local_buffer_pong_1);
            pong_size[2] =lz77_data_s2m(lz77_data_in[2], lz77_eos_in[2], local_buffer_pong_2);
            pong_size[3] =lz77_data_s2m(lz77_data_in[3], lz77_eos_in[3], local_buffer_pong_3);
            pong_size[4] =lz77_data_s2m(lz77_data_in[4], lz77_eos_in[4], local_buffer_pong_4);
            pong_size[5] =lz77_data_s2m(lz77_data_in[5], lz77_eos_in[5], local_buffer_pong_5);
            pong_size[6] =lz77_data_s2m(lz77_data_in[6], lz77_eos_in[6], local_buffer_pong_6);
            pong_size[7] =lz77_data_s2m(lz77_data_in[7], lz77_eos_in[7], local_buffer_pong_7);
            pong_size[8] =lz77_data_s2m(lz77_data_in[8], lz77_eos_in[8], local_buffer_pong_8);
            pong_size[9] =lz77_data_s2m(lz77_data_in[9], lz77_eos_in[9], local_buffer_pong_9);
            pong_size[10] =lz77_data_s2m(lz77_data_in[10], lz77_eos_in[10], local_buffer_pong_10);
            pong_size[11] =lz77_data_s2m(lz77_data_in[11], lz77_eos_in[11], local_buffer_pong_11);
            lz77_data_m2s(ping_size[0], lz77_data_out[0], local_buffer_ping_0);
            lz77_data_m2s(ping_size[1], lz77_data_out[1], local_buffer_ping_1);
            lz77_data_m2s(ping_size[2], lz77_data_out[2], local_buffer_ping_2);
            lz77_data_m2s(ping_size[3], lz77_data_out[3], local_buffer_ping_3);
            lz77_data_m2s(ping_size[4], lz77_data_out[4], local_buffer_ping_4);
            lz77_data_m2s(ping_size[5], lz77_data_out[5], local_buffer_ping_5);
            lz77_data_m2s(ping_size[6], lz77_data_out[6], local_buffer_ping_6);
            lz77_data_m2s(ping_size[7], lz77_data_out[7], local_buffer_ping_7);
            lz77_data_m2s(ping_size[8], lz77_data_out[8], local_buffer_ping_8);
            lz77_data_m2s(ping_size[9], lz77_data_out[9], local_buffer_ping_9);
            lz77_data_m2s(ping_size[10], lz77_data_out[10], local_buffer_ping_10);
            lz77_data_m2s(ping_size[11], lz77_data_out[11], local_buffer_ping_11);
            
        }
        */


        //#endif
        //    block_idx++;
            compressed_size_rd_wr(lz77_compressed_size_in,lz77_compressed_size_out);
    }
        //    printf(" rd pong \n");
        //    if(block_idx %2 == 1){
        //    lz77_data_m2s(ping_size[0], lz77_data_out[0], local_buffer_ping_0);
        //    lz77_data_m2s(ping_size[1], lz77_data_out[1], local_buffer_ping_1);
        //    lz77_data_m2s(ping_size[2], lz77_data_out[2], local_buffer_ping_2);
        //    lz77_data_m2s(ping_size[3], lz77_data_out[3], local_buffer_ping_3);
        //    lz77_data_m2s(ping_size[4], lz77_data_out[4], local_buffer_ping_4);
        //    lz77_data_m2s(ping_size[5], lz77_data_out[5], local_buffer_ping_5);
        //    lz77_data_m2s(ping_size[6], lz77_data_out[6], local_buffer_ping_6);
        //    lz77_data_m2s(ping_size[7], lz77_data_out[7], local_buffer_ping_7);
        //    lz77_data_m2s(ping_size[8], lz77_data_out[8], local_buffer_ping_8);
        //    lz77_data_m2s(ping_size[9], lz77_data_out[9], local_buffer_ping_9);
        //    lz77_data_m2s(ping_size[10], lz77_data_out[10], local_buffer_ping_10);
        //    lz77_data_m2s(ping_size[11], lz77_data_out[11], local_buffer_ping_11);
        //    }
        //    else{
        //    lz77_data_m2s(pong_size[0], lz77_data_out[0], local_buffer_pong_0);
        //    lz77_data_m2s(pong_size[1], lz77_data_out[1], local_buffer_pong_1);
        //    lz77_data_m2s(pong_size[2], lz77_data_out[2], local_buffer_pong_2);
        //    lz77_data_m2s(pong_size[3], lz77_data_out[3], local_buffer_pong_3);
        //    lz77_data_m2s(pong_size[4], lz77_data_out[4], local_buffer_pong_4);
        //    lz77_data_m2s(pong_size[5], lz77_data_out[5], local_buffer_pong_5);
        //    lz77_data_m2s(pong_size[6], lz77_data_out[6], local_buffer_pong_6);
        //    lz77_data_m2s(pong_size[7], lz77_data_out[7], local_buffer_pong_7);
        //    lz77_data_m2s(pong_size[8], lz77_data_out[8], local_buffer_pong_8);
        //    lz77_data_m2s(pong_size[9], lz77_data_out[9], local_buffer_pong_9);
        //    lz77_data_m2s(pong_size[10], lz77_data_out[10], local_buffer_pong_10);
        //    lz77_data_m2s(pong_size[11], lz77_data_out[11], local_buffer_pong_11);
        //    }
}
//}
