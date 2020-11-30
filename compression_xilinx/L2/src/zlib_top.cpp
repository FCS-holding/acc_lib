//#include "zlib_top.hpp"
#include "zlib_lz77_compress_mm.hpp"
#include "zlib_treegen_mm.hpp"
#include "zlib_huffman_enc_mm.hpp"
#include "zlib_s2_pingpong_buffer.hpp"
#include "zlib_config.hpp"
#define HOST_BUFFER_SIZE 32*1024/16
#include "stdio.h"
extern "C" {

void zlib_top(
    xf::compression::uintMemWidth_t* in,
    uint32_t in_block_size[PARALLEL_BLOCK],
    uint32_t block_size_in_kb,
    uint32_t input_size,
    uint32_t blocks_per_chunk,
    xf::compression::uintMemWidth_t* out,
    uint32_t  compressd_size[PARALLEL_BLOCK]
    ) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = in_block_size offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = compressd_size offset = slave bundle = gmem0

#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = in_block_size bundle = control
#pragma HLS INTERFACE s_axilite port = block_size_in_kb bundle = control
#pragma HLS INTERFACE s_axilite port = input_size bundle = control
#pragma HLS INTERFACE s_axilite port = blocks_per_chunk bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = compressd_size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    //printf("size = %d\n", input_size);

    uint32_t lz77_out_size[PARALLEL_BLOCK];
    #pragma HLS ARRAY_PARTITION variable = lz77_out_size dim = 0 complete
    hls::stream<ap_uint<32> > lz77_out[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = lz77_out          depth = 1024
    //#pragma HLS RESOURCE variable = lz77_out        core = FIFO_SRL
    hls::stream<bool> lz77_out_eos[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = lz77_out_eos      depth = 1024
    hls::stream<uint32_t> dyn_ltree_freq[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = dyn_ltree_freq    depth = 2048
    hls::stream<uint32_t> dyn_dtree_freq[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = dyn_dtree_freq    depth = 128
    hls::stream<uint32_t> dyn_ltree_codes[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = dyn_ltree_codes   depth = 2048
    hls::stream<uint32_t> dyn_dtree_codes[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = dyn_dtree_codes   depth = 128
    hls::stream<uint32_t> dyn_bltree_codes[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = dyn_bltree_codes  depth = 128
    hls::stream<uint32_t> dyn_ltree_blen[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = dyn_ltree_blen    depth = 2048
    hls::stream<uint32_t> dyn_dtree_blen[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = dyn_dtree_blen    depth = 128
    hls::stream<uint32_t> dyn_bltree_blen[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = dyn_bltree_blen   depth = 128
    hls::stream<uint32_t> max_codes[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = max_codes         depth = 128
    //hls::stream<ap_uint<32> > huffman_in[PARALLEL_BLOCK];
    //#pragma HLS STREAM variable = huffman_in        depth = 2
    ap_uint<32> local_buffer[PARALLEL_BLOCK][65536];
    #pragma HLS RESOURCE variable = local_buffer core = XPM_MEMORY uram
    #pragma HLS ARRAY_PARTITION variable = local_buffer dim = 1 complete
    uint32_t out_size[PARALLEL_BLOCK];
    #pragma HLS ARRAY_PARTITION variable = out_size dim = 0 complete
    
    xilLz77Compress(
        in,
        //lz77_out,
        //lz77_out_eos,
        local_buffer,
        out_size,
        lz77_out_size,
        //compressd_size,
        in_block_size,
        dyn_ltree_freq,
        dyn_dtree_freq,
        block_size_in_kb,
        input_size);

    xilTreegenKernel(
        dyn_ltree_freq,
        dyn_dtree_freq,
        dyn_ltree_codes,
        dyn_dtree_codes,
        dyn_bltree_codes,
        dyn_ltree_blen,
        dyn_dtree_blen,
        dyn_bltree_blen,
        max_codes,
        block_size_in_kb,
        input_size,
        blocks_per_chunk);
//=====================================
//add a local buffer for lz77 output
//transfer to stream to huffman module
//=====================================


//for(int i =0; i < PARALLEL_BLOCK; i++){
//#pragma HLS unroll
//    uint32_t cnt = 0;
//    for(bool eos = lz77_out_eos[i].read(); eos == false; eos = lz77_out_eos[i].read())
//    {
//    #pragma HLS pipeline
//        local_buffer[i][cnt++] = lz77_out[i].read();   
//    }
//    out_size[i] = cnt;
//    ap_uint<32> tmp = lz77_out[i].read();
//}



//#pragma HLS dataflow
//for(int i = 0; i < PARALLEL_BLOCK; i++){
//#pragma HLS unroll
//    for(int j = 0; j < out_size[i]; j++){
//    #pragma HLS pipeline
//        huffman_in[i] << local_buffer[i][j];;
//        }
//    }

//====================================
//finish transfer
//====================================

    xilHuffmanKernel(
        local_buffer,
        out_size,
        //huffman_in,
        //lz77_out,
        out,
        lz77_out_size,
        //compressd_size,
        //out_size,
        compressd_size,
        dyn_ltree_codes,
        dyn_dtree_codes,
        dyn_bltree_codes,
        dyn_ltree_blen,
        dyn_dtree_blen,
        dyn_bltree_blen,
        max_codes,
        block_size_in_kb,
        input_size);
        
    }
}
