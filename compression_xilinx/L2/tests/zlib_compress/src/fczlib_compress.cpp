#include <stdlib.h>  
#include "zlib.hpp"
//#include "license.h" 
//#ifdef __cplusplus
//extern "C"{
//#endif
void fczlib_init( const char* bin_name, uint32_t file_size){
    uint32_t compress_block_size;
    if (file_size <= 1048576*3 ) 
        compress_block_size = 32;
    else if(file_size > 1024*1024*3 && file_size <= 1024*1024*16)
        compress_block_size = 64;
    else if(file_size > 1048576*16 && file_size <= 240*1024*1024)//3-240MB
        compress_block_size = 256;
    else
        compress_block_size = 1024;
    
    //std::string compress_bin = "kernel_top.xclbin";
    std::string compress_bin = &bin_name[0];
    //Feature feature = FALCON_FCZIP;
    //fc_license_init(); 
    //fc_license_checkout(feature, 1);
    //fc_license_checkin(feature);
    xil_zlib(compress_bin,compress_block_size);
}
void fczlib_compress(
   uint8_t * in,
   uint32_t * in_size,
   uint8_t * out,
   uint32_t * out_size,
   uint8_t  compress_num,//max 1024
   uint32_t compress_config
){
    //printf("file num %d \n", compress_file_num);
    //printf("file size %d \n", in_file_size[0]);
    std::chrono::duration<double, std::nano> compress_API_time_ns_1(0);
    uint32_t * data_block_size = (uint32_t *)malloc(sizeof(uint32_t)*1024*128);
    uint32_t * compress_size = (uint32_t *)malloc(sizeof(uint32_t)*1024*128);
    uint32_t block_num_file[1024];
    uint32_t data_block_index = 0;
    uint32_t compress_block_size = 1024;
    uint32_t total_file_size = 0;
    for(uint16_t file_cnt = 0; file_cnt < compress_num; file_cnt++){
        uint32_t tmp_file_size = in_size[file_cnt];
        total_file_size += tmp_file_size;
        block_num_file[file_cnt] = (tmp_file_size - 1)/32768 + 1;
        for(uint32_t idx = 0; idx < tmp_file_size; idx += 32768){
            if(idx + 32768 > tmp_file_size){
                data_block_size[data_block_index] = tmp_file_size - idx;
                }
            else{
                data_block_size[data_block_index] = 32768;
                }
            data_block_index++;
            }
        }

    if (total_file_size <= 1048576*3 ) 
        compress_block_size = 32;
    else if(total_file_size > 1024*1024*3 && total_file_size <= 1024*1024*16)
        compress_block_size = 64;
    else if(total_file_size > 1048576*16 && total_file_size <= 240*1024*1024)//3-240MB
        compress_block_size = 256;
    else
        compress_block_size = 1024;
    auto compress_API_start = std::chrono::high_resolution_clock::now();
    compress(in, out, compress_block_size, data_block_index, data_block_size, compress_size);
    auto compress_API_end = std::chrono::high_resolution_clock::now();
    uint32_t compress_block_index = 0;
    for(uint32_t x = 0; x < compress_num; x++){
        for(uint32_t t = 0; t < block_num_file[x]; t++){
            out_size[x] += compress_size[compress_block_index];
            compress_block_index++;
        } 
    }
    auto duration = std::chrono::duration<double, std::nano>(compress_API_end - compress_API_start);
    compress_API_time_ns_1 += duration;
    float throughput_in_mbps_1 = (float)total_file_size * 1000 / compress_API_time_ns_1.count();
    if(compress_config == 1){
    printf("E2E throughput: ");
    std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;
    printf(" MB/s");
    printf("\n");
    }
}
void fczlib_release(){
    xil_zlib_dec();
    }
//#ifdef __cplusplus
//}
//#endif
