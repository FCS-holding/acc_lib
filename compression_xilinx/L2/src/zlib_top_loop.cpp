//#include "zlib_top.hpp"
#include "zlib_lz77_compress_mm.hpp"
#include "zlib_treegen_mm.hpp"
#include "zlib_huffman_enc_mm.hpp"
#include "zlib_s2_pingpong_buffer.hpp"
#include "zlib_config.hpp"
#define HOST_BUFFER_SIZE 32*1024/16
#include "stdio.h"


#define KERNEL_BUF_SIZE 32768
#define PROCESS_BLOCK_SIZE 262144*32
#define BLOCK_SIZE_IN_KB 32
extern "C" {

void zlib_top(
    xf::compression::uintMemWidth_t* in,
    uint32_t* in_block_size,
    uint32_t block_size_in_kb,
    uint32_t input_size,
    uint32_t blocks_per_chunk,
    xf::compression::uintMemWidth_t* out,
    uint32_t* compressd_size
    ) {
#if PLATFORM == 250
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = in_block_size offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = compressd_size offset = slave bundle = gmem1
#else
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = in_block_size offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = compressd_size offset = slave bundle = gmem3

#endif
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = in_block_size bundle = control
#pragma HLS INTERFACE s_axilite port = block_size_in_kb bundle = control
#pragma HLS INTERFACE s_axilite port = input_size bundle = control
#pragma HLS INTERFACE s_axilite port = blocks_per_chunk bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = compressd_size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    printf("input size %d \n",input_size);
    
    hls::stream<uint32_t> lz77_out_size[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = lz77_out_size          depth = 16
    hls::stream<ap_uint<32> > lz77_out[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = lz77_out          depth = 16
    hls::stream<bool> lz77_out_eos[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = lz77_out_eos      depth = 16
    hls::stream<uint32_t> dyn_ltree_freq[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = dyn_ltree_freq    depth = 1024
    hls::stream<uint32_t> dyn_dtree_freq[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = dyn_dtree_freq    depth = 128
    hls::stream<uint16_t> dataStreamTree[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = dataStreamTree    depth = 1024
    hls::stream<uint8_t> dataStreamSize[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = dataStreamSize   depth = 1024
    hls::stream<ap_uint<32> > huffman_in[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = huffman_in        depth = 16
    hls::stream<uint32_t> huffman_in_size[PARALLEL_BLOCK];
    #pragma HLS STREAM variable = huffman_in_size        depth = 16
    #pragma HLS dataflow
    xilLz77Compress(
        in,
        lz77_out,
        lz77_out_eos,
        lz77_out_size,
        in_block_size,
        dyn_ltree_freq,
        dyn_dtree_freq,
        BLOCK_SIZE_IN_KB,
        input_size
        );
    
    xilTreegenKernel(
        dyn_ltree_freq,
        dyn_dtree_freq,
        dataStreamTree,
        dataStreamSize,
        //lz77_out,
        //lz77_out_eos,
        lz77_out_size,
        //huffman_in,
        huffman_in_size,

        BLOCK_SIZE_IN_KB,
        input_size,
        blocks_per_chunk);
    s2pp_buffer(
        lz77_out,
        lz77_out_eos,
        huffman_in,
        BLOCK_SIZE_IN_KB,
        input_size,
        blocks_per_chunk
        );


    xilHuffmanKernel(
        huffman_in,
        out,
        huffman_in_size,
        compressd_size,
        dataStreamTree,
        dataStreamSize,
        BLOCK_SIZE_IN_KB,
        input_size
        );
        
}
}
