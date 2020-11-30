#include "hls_stream.h"

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <ap_int.h>
void s2pp_buffer(
    hls::stream<ap_uint<32> > instream[PARALLEL_BLOCK],
    hls::stream<bool> instream_eos[PARALLEL_BLOCK],
    hls::stream<ap_uint<32> > outstream[PARALLEL_BLOCK],
    uint32_t noBlocks

){
//=====================================
//add a local buffer for lz77 output
//transfer to stream to huffman module
//=====================================
uint32_t out_idx[PARALLEL_BLOCK];
bool w_pingpong_flag = 0;
bool r_pingpong_flag = 0;
uint8_t used_cnt = 0;
bool instream_end_flag[PARALLEL_BLOCK];
uint32_t w_cnt = 0;
for(int i = 0; i < PARALLEL_BLOCK; i++){
    instream_end_flag[i] = 0; 
    }
ap_uint<32> local_buffer[PARALLEL_BLOCK][65536][2];
#pragma HLS RESOURCE variable = local_buffer core = XPM_MEMORY uram
for(int k = 0; k < noBlocks; k = k + PARALLEL_BLOCK){    
    //printf("s2pp loop k %d \n",k);
    //printf("start to read data from stream \n");
    //printf("w_pingpong_flag %d \n",w_pingpong_flag);
    for(int i =0; i < PARALLEL_BLOCK; i++){
        w_cnt = 0;
        for(bool eos = instream_eos[i].read(); eos == false; eos = instream_eos[i].read())
        {
            local_buffer[i][w_cnt++][w_pingpong_flag] = instream[i].read();    
        }
        instream_end_flag[i] = 1;
        ap_uint<32> tmp = instream[i].read();
        out_idx[i] = w_cnt;
        }
    unsigned char tmp_cnt = 0;
    for(int i = 0; i < PARALLEL_BLOCK; i++){
        if(instream_end_flag[i] == 1)
        {
            tmp_cnt = tmp_cnt + 1;    
        }
    }
    if(tmp_cnt == 8){
        if(used_cnt < 2)
            used_cnt = used_cnt + 1;
        w_pingpong_flag = ~w_pingpong_flag;
    }
    //printf("w_pingpong_flag %d \n",w_pingpong_flag);
    //printf("start to write data to stream \n");
    //printf("r_pingpong_flag %d \n",r_pingpong_flag);
    //for(int i = 0; i < PARALLEL_BLOCK; i++){
    //    printf("out idx %d %d \n",i,out_idx[i]);
    //    }
    for(int i = 0; i < PARALLEL_BLOCK; i++){
        for(int j = 0; j < out_idx[i]; j++){
            outstream[i] << local_buffer[i][j][r_pingpong_flag];
            }
        }
    used_cnt = used_cnt - 1;
    r_pingpong_flag = ~r_pingpong_flag;
}
//====================================
//finish transfer
//====================================
/*
ap_uint<32> local_buffer[PARALLEL_BLOCK][65536];
#pragma HLS RESOURCE variable = local_buffer core = XPM_MEMORY uram
uint32_t out_size[PARALLEL_BLOCK];

for(int i =0; i < PARALLEL_BLOCK; i++){
    uint32_t cnt = 0;
    for(bool eos = instream_eos[i].read(); eos == false; eos = instream_eos[i].read())
    {
        local_buffer[i][cnt++] = instream[i].read();    
    }
    out_size[i] = cnt;
    ap_uint<32> tmp = instream[i].read();
    }
for(int i = 0; i < PARALLEL_BLOCK; i++){
    for(int j = 0; j < out_size[i]; j++){
        outstream[i] << local_buffer[i][j];;
        }
    }
*/
}
