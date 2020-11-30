#include "hls_stream.h"

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <ap_int.h>
uint32_t lz77_data_s2m(
    hls::stream<ap_uint<32> >& lz77_in,
    hls::stream<bool>& lz77_in_eos,
    ap_uint<64> local_buffer[32768]
    //hls::stream<uint32_t>& output_size
){
        
    uint32_t data_size;
    uint32_t cnt = 0;
    uint32_t data_cnt = 0;
    bool done = false;
    uint8_t d_cnt = 0;
    ap_uint<64> tmp_data_in;
    uint32_t addr_offset = 0;
    for(;done == false;){
    #pragma HLS PIPELINE II = 1
        if(!lz77_in_eos.empty()){
            bool eos_flag = lz77_in_eos.read();
            uint32_t tmp = lz77_in.read();
            if(eos_flag){
                if(d_cnt == 1){
                    tmp_data_in.range(31,0) = 0;
                    local_buffer[cnt + addr_offset] = tmp_data_in;
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
                    local_buffer[cnt + addr_offset] = tmp_data_in;
                    cnt++;
                    data_cnt++;
                    d_cnt = 0;
                    }
                }
            }
        }
    //output_size << data_cnt;
    printf("data_cnt %d \n",data_cnt);
    return data_cnt;
}
void lz77_data_m2s(
    uint32_t input_size,
    hls::stream<ap_uint<32> >&huffman_data_out,
    ap_uint<64> local_buffer[32768]
){
    uint32_t data_size;
    //data_size = input_size.read();
    data_size = input_size;
    printf("input size %d \n", data_size);
    if(data_size == 0) return;
    uint32_t addr_offset = 0;
    for(uint32_t i = 0; i < data_size; i++){
#pragma HLS pipeline II=1
        if(i%2 == 0){
            huffman_data_out << local_buffer[i/2].range(63,32);  
            uint32_t tmp = local_buffer[i/2].range(63,32);
            }
        else{
            huffman_data_out << local_buffer[i/2].range(31,0);   
            uint32_t tmp = local_buffer[i/2].range(31,0);
            }
        }
}
void lz77_data_buffer_ch(
    uint32_t no_blocks,
    hls::stream<ap_uint<32> >& lz77_in,
    hls::stream<bool>& lz77_in_eos,
    hls::stream<ap_uint<32> >& huffman_data_out,
    ap_uint<64> local_buffer_ping[16384],
    ap_uint<64> local_buffer_pong[16384],
    uint32_t ping_size[1],
    uint32_t pong_size[1]
){


    if(no_blocks %2 == 0){
        ping_size[0] =lz77_data_s2m(lz77_in, lz77_in_eos, local_buffer_ping);
        lz77_data_m2s(pong_size[0], huffman_data_out, local_buffer_pong);
        }
    else{
        pong_size[0] =lz77_data_s2m(lz77_in, lz77_in_eos, local_buffer_pong);
        lz77_data_m2s(ping_size[0], huffman_data_out, local_buffer_ping);
        }
//    ap_uint<64> local_buffer[32768];
//    #pragma HLS RESOURCE variable = local_buffer core = XPM_MEMORY uram
//    hls::stream<uint32_t> data_size;
//    #pragma HLS STREAM variable = data_size          depth = 16
//    for(int i = 0; i < no_blocks; i++){
    //#pragma HLS dataflow
        //lz77_data_s2m(i,lz77_in, lz77_in_eos, local_buffer, data_size);
        //lz77_data_m2s(i,data_size, huffman_data_out, local_buffer);
//    }
//    uint32_t data_size;
//    uint32_t cnt = 0;
//    uint32_t data_cnt = 0;
//    bool done = false;
//    uint8_t w_pipo_flag = block_idx % 2;
//    uint8_t r_pipo_flag = block_idx % 2;
//    uint8_t d_cnt = 0;
//    ap_uint<64> tmp_data_in;
//    for(;done == false;){
//    #pragma HLS PIPELINE II = 1
//        if(!lz77_in_eos.empty()){
//            bool eos_flag = lz77_in_eos.read();
//            uint32_t tmp = lz77_in.read();
//            if(eos_flag){
//                if(d_cnt == 1){
//                    tmp_data_in.range(31,0) = 0;
//                    local_buffer[w_pipo_flag][cnt] = tmp_data_in;
//                    //data_cnt++;
//                    cnt++;
//                    }
//                done = true;
//                }
//            else{
//                if(d_cnt == 0){
//                    tmp_data_in.range(31,0) = tmp;
//                    tmp_data_in = tmp_data_in << 32;
//                    d_cnt = d_cnt + 1;
//                    data_cnt++;
//                    }
//                else{
//                    tmp_data_in.range(31,0) = tmp;
//                    local_buffer[w_pipo_flag][cnt] = tmp_data_in;
//                    cnt++;
//                    data_cnt++;
//                    d_cnt = 0;
//                    }
//                }
//            }
//        }
//    data_size = data_cnt;
//    for(uint32_t i = 0; i < data_size; i++){
//#pragma HLS pipeline II=1
//        if(i%2 == 0){
//            huffman_data_out << local_buffer[r_pipo_flag][i/2].range(63,32);  
//            uint32_t tmp = local_buffer[r_pipo_flag][i/2].range(63,32);
//            }
//        else{
//            huffman_data_out << local_buffer[r_pipo_flag][i/2].range(31,0);   
//            uint32_t tmp = local_buffer[r_pipo_flag][i/2].range(31,0);
//            }
//        }
    }
void s2pp_buffer(
    hls::stream<ap_uint<32> > instream[PARALLEL_BLOCK],
    hls::stream<bool> instream_eos[PARALLEL_BLOCK],
    hls::stream<ap_uint<32> > outstream[PARALLEL_BLOCK],
    uint32_t block_size_in_kb,
    uint32_t input_size,
    uint32_t blocks_per_chunk

){

    int block_length = block_size_in_kb * 1024;
    int no_blocks = (input_size - 1) / block_length + 1;
    uint32_t block_idx = 0;

    uint32_t ping_size[PARALLEL_BLOCK];
#pragma HLS array_partition variable=ping_size dim=1 complete
    uint32_t pong_size[PARALLEL_BLOCK];
#pragma HLS array_partition variable=pong_size dim=1 complete
    for(int j = 0; j < PARALLEL_BLOCK; j++){
    #pragma HLS unroll
        ping_size[j] = 0;
        pong_size[j] = 0;
    }

    ap_uint<64> local_buffer_ping_0[16384];
    #pragma HLS RESOURCE variable = local_buffer_ping_0 core = XPM_MEMORY uram
    ap_uint<64> local_buffer_pong_0[16384];
    #pragma HLS RESOURCE variable = local_buffer_pong_0 core = XPM_MEMORY uram
    ap_uint<64> local_buffer_ping_1[16384];
    #pragma HLS RESOURCE variable = local_buffer_ping_1 core = XPM_MEMORY uram
    ap_uint<64> local_buffer_pong_1[16384];
    #pragma HLS RESOURCE variable = local_buffer_pong_1 core = XPM_MEMORY uram
    ap_uint<64> local_buffer_ping_2[16384];
    #pragma HLS RESOURCE variable = local_buffer_ping_2 core = XPM_MEMORY uram
    ap_uint<64> local_buffer_pong_2[16384];
    #pragma HLS RESOURCE variable = local_buffer_pong_2 core = XPM_MEMORY uram
    ap_uint<64> local_buffer_ping_3[16384];
    #pragma HLS RESOURCE variable = local_buffer_ping_3 core = XPM_MEMORY uram
    ap_uint<64> local_buffer_pong_3[16384];
    #pragma HLS RESOURCE variable = local_buffer_pong_3 core = XPM_MEMORY uram
    ap_uint<64> local_buffer_ping_4[16384];
    #pragma HLS RESOURCE variable = local_buffer_ping_4 core = XPM_MEMORY uram
    ap_uint<64> local_buffer_pong_4[16384];
    #pragma HLS RESOURCE variable = local_buffer_pong_4 core = XPM_MEMORY uram
    ap_uint<64> local_buffer_ping_5[16384];
    #pragma HLS RESOURCE variable = local_buffer_ping_5 core = XPM_MEMORY uram
    ap_uint<64> local_buffer_pong_5[16384];
    #pragma HLS RESOURCE variable = local_buffer_pong_5 core = XPM_MEMORY uram
    ap_uint<64> local_buffer_ping_6[16384];
    #pragma HLS RESOURCE variable = local_buffer_ping_6 core = XPM_MEMORY uram
    ap_uint<64> local_buffer_pong_6[16384];
    #pragma HLS RESOURCE variable = local_buffer_pong_6 core = XPM_MEMORY uram
    ap_uint<64> local_buffer_ping_7[16384];
    #pragma HLS RESOURCE variable = local_buffer_ping_7 core = XPM_MEMORY uram
    ap_uint<64> local_buffer_pong_7[16384];
    #pragma HLS RESOURCE variable = local_buffer_pong_7 core = XPM_MEMORY uram
    //ap_uint<64> local_buffer_ping_8[16384];
    //#pragma HLS RESOURCE variable = local_buffer_ping_8 core = XPM_MEMORY uram
    //ap_uint<64> local_buffer_pong_8[16384];
    //#pragma HLS RESOURCE variable = local_buffer_pong_8 core = XPM_MEMORY uram
    //ap_uint<64> local_buffer_ping_9[16384];
    //#pragma HLS RESOURCE variable = local_buffer_ping_9 core = XPM_MEMORY uram
    //ap_uint<64> local_buffer_pong_9[16384];
    //#pragma HLS RESOURCE variable = local_buffer_pong_9 core = XPM_MEMORY uram
    //ap_uint<64> local_buffer_ping_10[16384];
    //#pragma HLS RESOURCE variable = local_buffer_ping_10 core = XPM_MEMORY uram
    //ap_uint<64> local_buffer_pong_10[16384];
    //#pragma HLS RESOURCE variable = local_buffer_pong_10 core = XPM_MEMORY uram
    //ap_uint<64> local_buffer_ping_11[16384];
    //#pragma HLS RESOURCE variable = local_buffer_ping_11 core = XPM_MEMORY uram
    //ap_uint<64> local_buffer_pong_11[16384];
    //#pragma HLS RESOURCE variable = local_buffer_pong_11 core = XPM_MEMORY uram

    for(uint32_t b_idx = 0; b_idx < no_blocks; b_idx = b_idx + PARALLEL_BLOCK){
        #if PARALLEL_BLOCK==8
            lz77_data_buffer_ch(block_idx,instream[0],instream_eos[0],outstream[0],local_buffer_ping_0,local_buffer_pong_0,&ping_size[0],&pong_size[0]);
            lz77_data_buffer_ch(block_idx,instream[1],instream_eos[1],outstream[1],local_buffer_ping_1,local_buffer_pong_1,&ping_size[1],&pong_size[1]);
            lz77_data_buffer_ch(block_idx,instream[2],instream_eos[2],outstream[2],local_buffer_ping_2,local_buffer_pong_2,&ping_size[2],&pong_size[2]);
            lz77_data_buffer_ch(block_idx,instream[3],instream_eos[3],outstream[3],local_buffer_ping_3,local_buffer_pong_3,&ping_size[3],&pong_size[3]);
            lz77_data_buffer_ch(block_idx,instream[4],instream_eos[4],outstream[4],local_buffer_ping_4,local_buffer_pong_4,&ping_size[4],&pong_size[4]);
            lz77_data_buffer_ch(block_idx,instream[5],instream_eos[5],outstream[5],local_buffer_ping_5,local_buffer_pong_5,&ping_size[5],&pong_size[5]);
            lz77_data_buffer_ch(block_idx,instream[6],instream_eos[6],outstream[6],local_buffer_ping_6,local_buffer_pong_6,&ping_size[6],&pong_size[6]);
            lz77_data_buffer_ch(block_idx,instream[7],instream_eos[7],outstream[7],local_buffer_ping_7,local_buffer_pong_7,&ping_size[7],&pong_size[7]);
        #else
            lz77_data_buffer_ch(block_idx,instream[0],instream_eos[0],outstream[0],local_buffer_ping_0,local_buffer_pong_0,&ping_size[0],&pong_size[0]);
            lz77_data_buffer_ch(block_idx,instream[1],instream_eos[1],outstream[1],local_buffer_ping_1,local_buffer_pong_1,&ping_size[1],&pong_size[1]);
            lz77_data_buffer_ch(block_idx,instream[2],instream_eos[2],outstream[2],local_buffer_ping_2,local_buffer_pong_2,&ping_size[2],&pong_size[2]);
            lz77_data_buffer_ch(block_idx,instream[3],instream_eos[3],outstream[3],local_buffer_ping_3,local_buffer_pong_3,&ping_size[3],&pong_size[3]);
            lz77_data_buffer_ch(block_idx,instream[4],instream_eos[4],outstream[4],local_buffer_ping_4,local_buffer_pong_4,&ping_size[4],&pong_size[4]);
            lz77_data_buffer_ch(block_idx,instream[5],instream_eos[5],outstream[5],local_buffer_ping_5,local_buffer_pong_5,&ping_size[5],&pong_size[5]);
            lz77_data_buffer_ch(block_idx,instream[6],instream_eos[6],outstream[6],local_buffer_ping_6,local_buffer_pong_6,&ping_size[6],&pong_size[6]);
            lz77_data_buffer_ch(block_idx,instream[7],instream_eos[7],outstream[7],local_buffer_ping_7,local_buffer_pong_7,&ping_size[7],&pong_size[7]);
            lz77_data_buffer_ch(block_idx,instream[8],instream_eos[8],outstream[8],local_buffer_ping_8,local_buffer_pong_8,&ping_size[8],&pong_size[8]);
            lz77_data_buffer_ch(block_idx,instream[9],instream_eos[9],outstream[9],local_buffer_ping_9,local_buffer_pong_9,&ping_size[9],&pong_size[9]);
            lz77_data_buffer_ch(block_idx,instream[10],instream_eos[10],outstream[10],local_buffer_ping_10,local_buffer_pong_10,&ping_size[10],&pong_size[10]);
            lz77_data_buffer_ch(block_idx,instream[11],instream_eos[11],outstream[11],local_buffer_ping_11,local_buffer_pong_11,&ping_size[11],&pong_size[11]);
       #endif 
       block_idx++;
        
        
    }
            if(block_idx %2 == 1){
            lz77_data_m2s(ping_size[0], outstream[0], local_buffer_ping_0);
            lz77_data_m2s(ping_size[1], outstream[1], local_buffer_ping_1);
            lz77_data_m2s(ping_size[2], outstream[2], local_buffer_ping_2);
            lz77_data_m2s(ping_size[3], outstream[3], local_buffer_ping_3);
            lz77_data_m2s(ping_size[4], outstream[4], local_buffer_ping_4);
            lz77_data_m2s(ping_size[5], outstream[5], local_buffer_ping_5);
            lz77_data_m2s(ping_size[6], outstream[6], local_buffer_ping_6);
            lz77_data_m2s(ping_size[7], outstream[7], local_buffer_ping_7);
            //lz77_data_m2s(ping_size[8], outstream[8], local_buffer_ping_8);
            //lz77_data_m2s(ping_size[9], outstream[9], local_buffer_ping_9);
            //lz77_data_m2s(ping_size[10],outstream[10], local_buffer_ping_10);
            //lz77_data_m2s(ping_size[11],outstream[11], local_buffer_ping_11);
            }
            else{
            lz77_data_m2s(pong_size[0], outstream[0], local_buffer_pong_0);
            lz77_data_m2s(pong_size[1], outstream[1], local_buffer_pong_1);
            lz77_data_m2s(pong_size[2], outstream[2], local_buffer_pong_2);
            lz77_data_m2s(pong_size[3], outstream[3], local_buffer_pong_3);
            lz77_data_m2s(pong_size[4], outstream[4], local_buffer_pong_4);
            lz77_data_m2s(pong_size[5], outstream[5], local_buffer_pong_5);
            lz77_data_m2s(pong_size[6], outstream[6], local_buffer_pong_6);
            lz77_data_m2s(pong_size[7], outstream[7], local_buffer_pong_7);
            //lz77_data_m2s(pong_size[8], outstream[8], local_buffer_pong_8);
            //lz77_data_m2s(pong_size[9], outstream[9], local_buffer_pong_9);
            //lz77_data_m2s(pong_size[10],outstream[10], local_buffer_pong_10);
            //lz77_data_m2s(pong_size[11],outstream[11], local_buffer_pong_11);
            }
}
