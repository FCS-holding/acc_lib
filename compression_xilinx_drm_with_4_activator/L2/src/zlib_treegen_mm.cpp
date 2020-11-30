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
                      hls::stream<uint32_t>& dyn_ltree_codes,  
                      hls::stream<uint32_t>& dyn_dtree_codes,  
                      hls::stream<uint32_t>& dyn_bltree_codes, 
                      hls::stream<uint32_t>& dyn_ltree_blen,   
                      hls::stream<uint32_t>& dyn_dtree_blen,   
                      hls::stream<uint32_t>& dyn_bltree_blen,  
                      hls::stream<uint32_t>& max_codes) {
    
    // Literal & Match Length content
    uint32_t lcl_ltree_freq[LTREE_SIZE];
    uint32_t lcl_ltree_codes[LTREE_SIZE];
    uint32_t lcl_ltree_blen[LTREE_SIZE];
    uint32_t lcl_ltree_root[LTREE_SIZE];

    // Distances content
    uint32_t lcl_dtree_freq[DTREE_SIZE];
    uint32_t lcl_dtree_codes[DTREE_SIZE];
    uint32_t lcl_dtree_blen[DTREE_SIZE];
    uint32_t lcl_dtree_root[DTREE_SIZE];

    // BL tree content
    uint32_t lcl_bltree_freq[BLTREE_SIZE];
    uint32_t lcl_bltree_codes[BLTREE_SIZE];
    uint32_t lcl_bltree_blen[BLTREE_SIZE];
    uint32_t lcl_bltree_root[BLTREE_SIZE];
    copy_ltree_freq:
        for (uint32_t ci = 0; ci < LTREE_SIZE; ++ci) {
#pragma HLS PIPELINE II = 1
            lcl_ltree_freq[ci] = dyn_ltree_freq.read();
        }
    copy_dtree_freq:
        for (uint32_t ci = 0; ci < DTREE_SIZE; ++ci) {
#pragma HLS PIPELINE II = 1
            lcl_dtree_freq[ci] = dyn_dtree_freq.read();
        }
        // Build Literal and Match length tree codes & bit lenghts
        uint32_t lit_max_code = xf::compression::huffConstructTree<LITERAL_CODES, MAX_BITS>(
            lcl_ltree_freq, lcl_ltree_codes, lcl_ltree_blen, lcl_ltree_root);

        // Build distances codes & bit lengths
        uint32_t dst_max_code = xf::compression::huffConstructTree<DISTANCE_CODES, MAX_BITS>(
            lcl_dtree_freq, lcl_dtree_codes, lcl_dtree_blen, lcl_dtree_root);

        uint8_t bitlen_vals[EXTRA_BLCODES] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
    bltree_init:
        for (int i = 0; i < BLTREE_SIZE; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 64
#pragma HLS PIPELINE II = 1
            lcl_bltree_freq[i] = 0;
        }

        uint32_t* tree_len = NULL;
        uint32_t max_code = 0;

        for (uint32_t pt = 0; pt < 2; pt++) {
            if (pt == 0) {
                tree_len = lcl_ltree_blen;
                max_code = lit_max_code;
            } else {
                tree_len = lcl_dtree_blen;
                max_code = dst_max_code;
            }

            int prevlen = -1;
            int curlen = 0;
            int count = 0;
            int max_count = 7;
            int min_count = 4;
            int nextlen = tree_len[0];

            if (nextlen == 0) {
                max_count = 138;
                min_count = 3;
            }

            tree_len[max_code + 1] = (uint16_t)0xffff;
        parse_tdata:
            for (uint32_t n = 0; n <= max_code; n++) {
#pragma HLS LOOP_TRIPCOUNT min = 286 max = 286
#pragma HLS PIPELINE II = 1
                curlen = nextlen;
                nextlen = tree_len[n + 1];

                if (++count < max_count && curlen == nextlen)
                    continue;
                else if (count < min_count)
                    lcl_bltree_freq[curlen] += count;
                else if (curlen != 0) {
                    if (curlen != prevlen) lcl_bltree_freq[curlen]++;
                    lcl_bltree_freq[REUSE_PREV_BLEN]++;
                } else if (count <= 10) {
                    lcl_bltree_freq[REUSE_ZERO_BLEN]++;
                } else {
                    lcl_bltree_freq[REUSE_ZERO_BLEN_7]++;
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

        /* Build the bit length tree */
        uint32_t max_blindex = xf::compression::huffConstructTree<BL_CODES, MAX_BL_BITS>(
            lcl_bltree_freq, lcl_bltree_codes, lcl_bltree_blen, lcl_bltree_root);

    bltree_blen:
        for (max_blindex = BL_CODES - 1; max_blindex >= 3; max_blindex--) {
            if (lcl_bltree_blen[bitlen_vals[max_blindex]] != 0) break;
        }

        max_codes << lit_max_code; // Writing to the output stream.
        max_codes << dst_max_code; // Writing to the output stream.
        max_codes << max_blindex; // Writing to the output stream.

    // Copy data back to ddr  -- Literals / MLs
    copy2ddr_ltree_codes:
        for (uint32_t ci = 0; ci < LTREE_SIZE; ++ci) {
#pragma HLS PIPELINE II = 1
            dyn_ltree_codes << lcl_ltree_codes[ci];
        }
    copy2ddr_ltree_blen:
        for (uint32_t ci = 0; ci < LTREE_SIZE; ++ci) {
#pragma HLS PIPELINE II = 1
            dyn_ltree_blen << lcl_ltree_blen[ci];
        }

    // Copy data back to ddr -- Distances
    copy2ddr_dtree_codes:
        for (uint32_t ci = 0; ci < DTREE_SIZE; ++ci) {
#pragma HLS PIPELINE II = 1
            dyn_dtree_codes << lcl_dtree_codes[ci];
        }
    copy2ddr_dtree_blen:
        for (uint32_t ci = 0; ci < DTREE_SIZE; ++ci) {
#pragma HLS PIPELINE II = 1
            dyn_dtree_blen << lcl_dtree_blen[ci]; 
        }

    // Copy data back to ddr -- Bit Lengths
    copy2ddr_bltree_codes:
        for (uint32_t ci = 0; ci < BLTREE_SIZE; ++ci) {
#pragma HLS PIPELINE II = 1
            dyn_bltree_codes << lcl_bltree_codes[ci];
        }
    copy2ddr_bltree_blen:
        for (uint32_t ci = 0; ci < BLTREE_SIZE; ++ci) {
#pragma HLS PIPELINE II = 1
            dyn_bltree_blen <<  lcl_bltree_blen[ci];
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
                      hls::stream<uint32_t> dyn_ltree_codes[PARALLEL_BLOCK],  
                      hls::stream<uint32_t> dyn_dtree_codes[PARALLEL_BLOCK],  
                      hls::stream<uint32_t> dyn_bltree_codes[PARALLEL_BLOCK], 
                      hls::stream<uint32_t> dyn_ltree_blen[PARALLEL_BLOCK],   
                      hls::stream<uint32_t> dyn_dtree_blen[PARALLEL_BLOCK],   
                      hls::stream<uint32_t> dyn_bltree_blen[PARALLEL_BLOCK],  
                      hls::stream<uint32_t> max_codes[PARALLEL_BLOCK],       

                      hls::stream<ap_uint<32> > lz77_data_in[PARALLEL_BLOCK],
                      hls::stream<bool> lz77_eos_in[PARALLEL_BLOCK],
                      hls::stream<uint32_t> lz77_compressed_size_in[PARALLEL_BLOCK],
                      hls::stream<ap_uint<32> > lz77_data_out[PARALLEL_BLOCK],
                      hls::stream<uint32_t> lz77_compressed_size_out[PARALLEL_BLOCK],

                      uint32_t block_size_in_kb,
                      uint32_t input_size,
                      uint32_t blocks_per_chunk) {
//#pragma HLS INTERFACE m_axi port = dyn_ltree_freq offset = slave bundle = gmem0
//#pragma HLS INTERFACE m_axi port = dyn_dtree_freq offset = slave bundle = gmem0
//#pragma HLS INTERFACE m_axi port = dyn_bltree_freq offset = slave bundle = gmem0
//#pragma HLS INTERFACE m_axi port = dyn_ltree_codes offset = slave bundle = gmem0
//#pragma HLS INTERFACE m_axi port = dyn_dtree_codes offset = slave bundle = gmem0
//#pragma HLS INTERFACE m_axi port = dyn_bltree_codes offset = slave bundle = gmem0
//#pragma HLS INTERFACE m_axi port = dyn_ltree_blen offset = slave bundle = gmem0
//#pragma HLS INTERFACE m_axi port = dyn_dtree_blen offset = slave bundle = gmem0
//#pragma HLS INTERFACE m_axi port = dyn_bltree_blen offset = slave bundle = gmem0
//#pragma HLS INTERFACE m_axi port = max_codes offset = slave bundle = gmem0
//
//#pragma HLS INTERFACE s_axilite port = dyn_ltree_freq bundle = control
//#pragma HLS INTERFACE s_axilite port = dyn_dtree_freq bundle = control
//#pragma HLS INTERFACE s_axilite port = dyn_bltree_freq bundle = control
//#pragma HLS INTERFACE s_axilite port = dyn_ltree_codes bundle = control
//#pragma HLS INTERFACE s_axilite port = dyn_dtree_codes bundle = control
//#pragma HLS INTERFACE s_axilite port = dyn_bltree_codes bundle = control
//#pragma HLS INTERFACE s_axilite port = dyn_ltree_blen bundle = control
//#pragma HLS INTERFACE s_axilite port = dyn_dtree_blen bundle = control
//#pragma HLS INTERFACE s_axilite port = dyn_bltree_blen bundle = control
//#pragma HLS INTERFACE s_axilite port = max_codes bundle = control
//#pragma HLS INTERFACE s_axilite port = block_size_in_kb bundle = control
//#pragma HLS INTERFACE s_axilite port = input_size bundle = control
//#pragma HLS INTERFACE s_axilite port = blocks_per_chunk bundle = control
//#pragma HLS INTERFACE s_axilite port = return bundle = control
    const uint32_t c_gmemBSize = 1024;
    //output_size[0] = input_size;
    int block_length = block_size_in_kb * 1024;
    int no_blocks = (input_size - 1) / block_length + 1;
//#pragma HLS STREAM variable =dyn_ltree_codes   depth = c_gmemBSize
//#pragma HLS STREAM variable =dyn_dtree_codes   depth = 128
//#pragma HLS STREAM variable =dyn_bltree_codes  depth = 128
//#pragma HLS STREAM variable =dyn_ltree_blen    depth = c_gmemBSize
//#pragma HLS STREAM variable =dyn_dtree_blen    depth = 128
//#pragma HLS STREAM variable =dyn_bltree_blen   depth = 128
//#pragma HLS STREAM variable =max_codes         depth = 16
#pragma HLS STREAM variable =dyn_ltree_codes   depth = 16
#pragma HLS STREAM variable =dyn_dtree_codes   depth = 16
#pragma HLS STREAM variable =dyn_bltree_codes  depth = 16
#pragma HLS STREAM variable =dyn_ltree_blen    depth = 16
#pragma HLS STREAM variable =dyn_dtree_blen    depth = 16
#pragma HLS STREAM variable =dyn_bltree_blen   depth = 16
#pragma HLS STREAM variable =max_codes         depth = 16
    uint32_t block_idx = 0;
    for(uint32_t b_idx = 0; b_idx < no_blocks; b_idx = b_idx + PARALLEL_BLOCK){
        printf("treegen start %d \n",b_idx);
        for (uint8_t core_idx = 0; core_idx < PARALLEL_BLOCK; core_idx++) {
        #pragma HLS UNROLL
        // Copy Frequencies of Literal, Match Length and Distances to local buffers
            treegen_simple_ch(
                dyn_ltree_freq[core_idx],
                dyn_dtree_freq[core_idx],
                dyn_ltree_codes[core_idx],  
                dyn_dtree_codes[core_idx],  
                dyn_bltree_codes[core_idx], 
                dyn_ltree_blen[core_idx],   
                dyn_dtree_blen[core_idx],   
                dyn_bltree_blen[core_idx],  
                max_codes[core_idx] 
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
