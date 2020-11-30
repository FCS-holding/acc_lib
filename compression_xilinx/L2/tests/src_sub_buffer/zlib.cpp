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
#include "zlib.hpp"
#include "libgen.h"
#include <thread>
#include <pthread.h>
#include <mutex>
#include <sched.h>
#include <stdio.h>
#include <sys/prctl.h>
#define FORMAT_0 31
#define FORMAT_1 139
#define VARIANT 8
#define REAL_CODE 8
#define OPCODE 3
#define CHUNK_16K 16384
#define MAX_FILE_NUM 1024
#define BLOCK_MEM_SIZE 32768
//HBM Banks requirements
#define MAX_HBM_BANKCOUNT 32
#define BANK_NAME(n) n | XCL_MEM_TOPOLOGY
const int bank[MAX_HBM_BANKCOUNT] = {
    BANK_NAME(0),  BANK_NAME(1),  BANK_NAME(2),  BANK_NAME(3),  BANK_NAME(4),
    BANK_NAME(5),  BANK_NAME(6),  BANK_NAME(7),  BANK_NAME(8),  BANK_NAME(9),
    BANK_NAME(10), BANK_NAME(11), BANK_NAME(12), BANK_NAME(13), BANK_NAME(14),
    BANK_NAME(15), BANK_NAME(16), BANK_NAME(17), BANK_NAME(18), BANK_NAME(19),
    BANK_NAME(20), BANK_NAME(21), BANK_NAME(22), BANK_NAME(23), BANK_NAME(24),
    BANK_NAME(25), BANK_NAME(26), BANK_NAME(27), BANK_NAME(28), BANK_NAME(29),
    BANK_NAME(30), BANK_NAME(31)};



std::mutex mu;
uint32_t get_file_size(std::ifstream& file) {
    file.seekg(0, file.end);
    uint32_t file_size = file.tellg();
    file.seekg(0, file.beg);
    return file_size;
}

void memcpy_t(uint8_t *a, uint8_t *b, uint32_t size){
    //mu.lock();
    cpu_set_t cpu_mask;
    CPU_ZERO(&cpu_mask);
    CPU_SET(1, &cpu_mask);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpu_mask);
    memcpy(a,b,size);
    //mu.unlock();
    }

void data_read_out(
        uint32_t *outIdx, 
        uint32_t cu, 
        uint32_t bf_idx,
        uint32_t cb_size,
        uint32_t bk_size, 
        uint8_t *out,
        uint8_t *in,
        uint32_t *cr_size_in,
        uint32_t *cr_size_out, 
        uint32_t *blocksPerChunk, 
        uint32_t *sizeOfChunk, 
        uint32_t *compress_size_index
        )
{
    cpu_set_t cpu_mask;
    CPU_ZERO(&cpu_mask);
    CPU_SET(31, &cpu_mask);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpu_mask);
    uint32_t index = 0;
    for (uint32_t bIdx = 0; bIdx < blocksPerChunk[bf_idx]; bIdx++, index += bk_size) {
        uint32_t block_size = bk_size;
        if (index + block_size > sizeOfChunk[bf_idx]) {
            block_size = sizeOfChunk[bf_idx] - index;
        }
        uint32_t no_sub_block = (block_size - 1) / 32768 + 1;
        for(uint32_t sub_idx = 0; sub_idx < no_sub_block; sub_idx++){
            uint32_t compressed_size = cr_size_in[sub_idx + bIdx * cb_size/32];
            std::memcpy(&out[*outIdx], &in[bIdx* bk_size*33/32 + sub_idx*32768*33/32],compressed_size);
            *outIdx += compressed_size;
            cr_size_out[(*compress_size_index)++]= compressed_size;
        }
    }
}
void zip(std::string& inFile_name, std::ofstream& outFile, uint8_t* zip_out, uint32_t enbytes) {
    outFile.put(120);
    outFile.put(1);
    outFile.write((char*)zip_out, enbytes);
    outFile.put(0);
    outFile.put(0);
    outFile.put(0);
    outFile.put(0);
    outFile.put(0);
}

uint32_t xil_zlib::compress_file(std::string& inFile_name, std::string& outFile_name, uint64_t input_size, uint32_t compress_block_size, uint32_t mod_sel) {
    std::chrono::duration<double, std::nano> compress_API_time_ns_1(0);
    //std::ifstream inFile(inFile_name.c_str(), std::ifstream::binary);
    //std::ofstream outFile(outFile_name.c_str(), std::ofstream::binary);
    std::ifstream infilelist(inFile_name.c_str(),std::ifstream::binary);
    std::string line;
    std::string file_in_name[MAX_FILE_NUM];
    std::string file_org_name[MAX_FILE_NUM];
    //uint32_t pid = getpid();
    //printf("current pid %d \n",pid);
    //printf("available cpu number %d \n",n);
    int file_num = 0;
    int file_block_num[MAX_FILE_NUM];//up to 1024 files
    uint32_t file_size_array[MAX_FILE_NUM];//up to 1024 files
    int tmp_file_size;
    uint32_t total_file_size = 0;
    uint32_t real_file_size = 0;
    uint32_t total_block_num = 0;
    uint32_t block_num_count = 0;
    uint32_t * data_block_size = (uint32_t *)malloc(sizeof(uint32_t)*1024*48);
    uint32_t * compress_size_1 = (uint32_t *)malloc(sizeof(uint32_t)*1024*48);
    uint32_t * compress_size_2 = (uint32_t *)malloc(sizeof(uint32_t)*1024*48);
    uint32_t * compress_size_3 = (uint32_t *)malloc(sizeof(uint32_t)*1024*48);
    uint32_t * compress_size_4 = (uint32_t *)malloc(sizeof(uint32_t)*1024*48);
    //parse the file list and statics all file size and block number
    if(mod_sel == 1){
        while(std::getline(infilelist, line)){
            std::ifstream inFile(line.c_str(), std::ifstream::binary);
            if(inFile){
                file_in_name[file_num] = line.c_str();
                char *path_buffer = const_cast<char *>(line.c_str());
                file_org_name[file_num] = basename(path_buffer);
                
                tmp_file_size = get_file_size(inFile);
                real_file_size += tmp_file_size;
                file_size_array[file_num] = tmp_file_size;
                int tmp_block_num = (tmp_file_size - 1) / BLOCK_MEM_SIZE + 1;
                for(int j = 0; j < tmp_block_num; j++){
                   int tmp_size = 0;
                   if(j == tmp_block_num - 1){
                       tmp_size = tmp_file_size - (tmp_block_num - 1)*BLOCK_MEM_SIZE;
                       }
                   else{
                       tmp_size = BLOCK_MEM_SIZE; 
                       }
                       data_block_size[j + block_num_count] = tmp_size;
                    }
                block_num_count += tmp_block_num;
                file_block_num[file_num] = tmp_block_num;
                total_file_size += tmp_block_num * BLOCK_MEM_SIZE;
                total_block_num += tmp_block_num;
                file_num++;
                inFile.close();
                if(file_num >= MAX_FILE_NUM){
                    printf("too many files \n");
                    break;
                    }
            }
        }
    }
    if(mod_sel == 0){
        total_file_size = input_size;
        }
    //if (!inFile) {
    //    std::cout << "Unable to open file";
    //    exit(1);
    //}

    //std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_in(input_size);
    uint32_t total_block_num_tmp = (total_file_size - 1)/ 32768 + 1;
    std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_format(total_block_num_tmp * 32768);
    //std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_format(total_file_size);
    //std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_out(input_size * 2);
    uint32_t tmp_output_file_size = (total_file_size/8)*9 + 1;
    //printf("output file size %d \n",tmp_output_file_size);
    std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_out_1((tmp_output_file_size -1) /4 + 1);
    std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_out_2((tmp_output_file_size -1)/4 + 1);
    std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_out_3((tmp_output_file_size -1)/4 + 1);
    std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_out_4((tmp_output_file_size -1)/4 + 1);
    uint32_t pre_file_block_num = 0;
    if(mod_sel == 1){
        for(int i = 0; i < file_num; i++){
            std::ifstream inFile(file_in_name[i], std::ifstream::binary);
            inFile.read((char *)(zlib_format.data() + pre_file_block_num), file_size_array[i]);
            pre_file_block_num += file_block_num[i] * BLOCK_MEM_SIZE;
            //if(i == 0){
            //    inFile.read((char *)zlib_format.data(), file_size_array[i]); 
            //}
            //else{
            //    inFile.read((char *)(zlib_format.data() + file_block_num[i-1]*BLOCK_MEM_SIZE), file_size_array[i]); 
            //}
            inFile.close();
            
        }
    }
    else{
        infilelist.read((char*)zlib_format.data(), input_size);
        total_file_size = input_size;
        tmp_file_size = input_size;
        real_file_size += tmp_file_size;
        file_size_array[file_num] = tmp_file_size;
        int tmp_block_num = (tmp_file_size - 1) / BLOCK_MEM_SIZE + 1;
        for(int j = 0; j < tmp_block_num; j++){
           int tmp_size = 0;
           if(j == tmp_block_num - 1){
               tmp_size = tmp_file_size - (tmp_block_num - 1)*BLOCK_MEM_SIZE;
               }
           else{
               tmp_size = BLOCK_MEM_SIZE; 
               }
               data_block_size[j + block_num_count] = tmp_size;
            }
        block_num_count += tmp_block_num;
        file_block_num[file_num] = tmp_block_num;
        //total_file_size += tmp_block_num * BLOCK_MEM_SIZE;
        total_block_num += tmp_block_num;
        file_num++;
    }
    uint32_t compress_block_size_t;
    if (total_file_size <= 1048576*3 ) 
        compress_block_size_t = 32;
    else if(total_file_size > 1024*1024*3 && total_file_size <= 1024*1024*16)
        compress_block_size_t = 64;
    else if(total_file_size > 1048576*16 && total_file_size <= 240*1024*1024)//3-240MB
        compress_block_size_t = 256;
    else
        compress_block_size_t = 1024;

    uint32_t sizeOfChunk[512];
    //uint32_t blocksPerChunk[8];
    uint32_t idx = 0;
    for(int i = 0; i < 512; i++){
        sizeOfChunk[i] = 0;
        }
    //uint32_t host_buffer_size = 12*32768;
    uint32_t host_buffer_size = PARALLEL_ENGINES*compress_block_size_t*1024;
    //uint8_t *in_data = zlib_in.data();
    uint8_t *in_data = zlib_format.data();
    uint32_t total_chunk = 0;
    for (uint32_t i = 0; i < total_file_size; i += host_buffer_size, idx++) {
        uint32_t chunk_size = host_buffer_size;
        if (chunk_size + i > input_size) {
            chunk_size = input_size - i;
        }
        sizeOfChunk[idx] = chunk_size;
    }
    total_chunk = idx;
    uint32_t no_chunk = 0;
    uint32_t CU_NUM = 1;
    if(total_chunk > 2)
        CU_NUM = 4;
    no_chunk = (total_chunk - 1)/8 + 1;
    //printf("no_chunk %d \n", no_chunk);
    //if(total_file_size <= 12*1024*1024){
    //    for(uint32_t n = 0; n < no_chunk; n++){
    //        for (uint32_t t = 0; t < 2; t++){ 
    //            for (uint32_t cu = 0; cu < CU_NUM; cu++) {
    //                if((t*CU_NUM + cu + n*8) < total_chunk) {
    //                    std::memcpy(h_buf_in[n][cu][t].data(), &in_data[(t*CU_NUM + cu + n*8) * host_buffer_size], sizeOfChunk[t*CU_NUM + cu + n*8]);
    //                    }
    //                }
    //            }
    //        }
    //}
    //TEST SPEED 
    //for(uint32_t flag = 0; flag < 2; flag++){
    //    for(uint32_t cu = 0; cu < 4; cu++){            
    //        std::memcpy(h_buf_in_s[cu][flag].data(), &in_data[(flag*4 + cu) * host_buffer_size], sizeOfChunk[flag*4 + cu]);
    //    }
    //}
    auto compress_API_start = std::chrono::high_resolution_clock::now();
    // zlib Compress
    //uint32_t enbytes = compress(zlib_in.data(), zlib_out.data(), input_size, compress_block_size);
    uint32_t enbytes = compress(zlib_format.data(), zlib_out_1.data(),zlib_out_2.data(),zlib_out_3.data(),zlib_out_4.data(), compress_block_size_t,
                                total_block_num,
                                data_block_size,
                                compress_size_1,
                                compress_size_2,
                                compress_size_3,
                                compress_size_4
                        );
    //uint32_t enbytes = 0;
    //for(uint32_t i = 0; i < 8; i++){
    //    printf("call times %d \n",i);
    //    //compress_size_1 = (uint32_t *)malloc(sizeof(uint32_t)*1024*48);
    //    //compress_size_2 = (uint32_t *)malloc(sizeof(uint32_t)*1024*48);
    //    //compress_size_3 = (uint32_t *)malloc(sizeof(uint32_t)*1024*48);
    //    //compress_size_4 = (uint32_t *)malloc(sizeof(uint32_t)*1024*48);
    //    //std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_format_t(total_block_num_tmp * 32768);
    //    //std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_out_1_t((tmp_output_file_size -1) /4 + 1);
    //    //std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_out_2_t((tmp_output_file_size -1) /4 + 1);
    //    //std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_out_3_t((tmp_output_file_size -1) /4 + 1);
    //    //std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_out_4_t((tmp_output_file_size -1) /4 + 1);
    //    enbytes = compress(zlib_format.data(), zlib_out_1.data(),zlib_out_2.data(),zlib_out_3.data(),zlib_out_4.data(), compress_block_size_t,
    //                            total_block_num,
    //                            data_block_size,
    //                            compress_size_1,
    //                            compress_size_2,
    //                            compress_size_3,
    //                            compress_size_4
    //                    );
    //    printf("finish \n");
    //}
    auto compress_API_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::nano>(compress_API_end - compress_API_start);
    compress_API_time_ns_1 += duration;
 
    //float throughput_in_mbps_1 = (float)input_size * 1000 / compress_API_time_ns_1.count();
    float throughput_in_mbps_1 = (float)real_file_size * 1000 / compress_API_time_ns_1.count();
    std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;
    printf("\n");
    //printf("exe time %f \n", compress_API_time_ns_1.count());
    int i = 0;
    uint32_t block_index = 0;
    uint32_t data_read_index = 0;
    for(i = 0; i < file_num; i++){
        std::string out_file_name;
        if(mod_sel == 1){
            out_file_name = file_in_name[i]; 
            if(!outFile_name.empty()){
                std::string tmp_file_name;
                tmp_file_name = outFile_name + "/" + file_org_name[i] + ".zlib";
                out_file_name = tmp_file_name;
                }
            else{
                out_file_name = out_file_name + ".zlib";
            }
        }
        else{
            out_file_name = outFile_name;
        }
        std::ofstream outFile(out_file_name, std::ofstream::binary);
        //write header
        outFile.put(120);
        outFile.put(1);
        //printf("file block num %d \n",file_block_num[i]);
        uint32_t total_chunk_tmp = (file_block_num[i] - 1)/(8*compress_block_size_t/32) + 1;
        uint32_t pe_chunk_num[4];
        if(total_chunk_tmp <= 4){
            for(int x = 0; x < 4; x++){
                if(x < total_chunk_tmp){
                    pe_chunk_num[x] = 1;
                    }
                else{
                    pe_chunk_num[x] = 0;
                    }
                }
            }
        else{
            int average_chunks_pe = total_chunk_tmp / 4;
            int remainder_chunks = total_chunk_tmp % 4;
            for(int x = 0; x < 4; x++){
                if(x < remainder_chunks){
                    pe_chunk_num[x] = average_chunks_pe + 1;
                    }
                else{
                    pe_chunk_num[x] = average_chunks_pe;
                    }
                } 
            }
        //for(int j = 0; j < file_block_num[i]; j++){
        //    outFile.write((char *)(zlib_out.data() + data_read_index),compress_size[block_index]);
        //    data_read_index += compress_size[block_index];
        //    block_index++;
        //}
        //printf("pe chunk tmp %d \n", pe_chunk_num[0]); 
        //printf("pe chunk tmp %d \n", pe_chunk_num[1]); 
        //printf("pe chunk tmp %d \n", pe_chunk_num[2]); 
        //printf("pe chunk tmp %d \n", pe_chunk_num[3]); 
        for(int j = 0; j < pe_chunk_num[0]*8*compress_block_size_t/32;j++){
            outFile.write((char *)(zlib_out_1.data() + data_read_index),compress_size_1[block_index]);
            data_read_index += compress_size_1[block_index];
            block_index++;
            }
        block_index = 0;
        data_read_index = 0;
        for(int j = 0; j < pe_chunk_num[1]*8*compress_block_size_t/32;j++){
            outFile.write((char *)(zlib_out_2.data() + data_read_index),compress_size_2[block_index]);
            data_read_index += compress_size_2[block_index];
            block_index++;
            }
        block_index = 0;
        data_read_index = 0;
        for(int j = 0; j < pe_chunk_num[2]*8*compress_block_size_t/32;j++){
            outFile.write((char *)(zlib_out_3.data() + data_read_index),compress_size_3[block_index]);
            data_read_index += compress_size_3[block_index];
            block_index++;
            }
        block_index = 0;
        data_read_index = 0;
        for(int j = 0; j < pe_chunk_num[3]*8*compress_block_size_t/32;j++){
            outFile.write((char *)(zlib_out_4.data() + data_read_index),compress_size_4[block_index]);
            data_read_index += compress_size_4[block_index];
            block_index++;
            }
        
        outFile.put(0x01);
        outFile.put(0x00);
        outFile.put(0x00);
        outFile.put(0xff);
        outFile.put(0xff);

        outFile.put(0);
        outFile.put(0);
        outFile.put(0);
        outFile.put(0);
        outFile.put(0);
        outFile.close();
        }
        //enbytes = data_read_index;
    // Pack zlib encoded stream .gz file
    //zip(inFile_name, outFile, zlib_out.data(), enbytes);
    free(compress_size_1);
    free(compress_size_2);
    free(compress_size_3);
    free(compress_size_4);
    free(data_block_size);
    //delete(zlib_format.data());
    //delete(zlib_out_1.data());
    //delete(zlib_out_2.data());
    //delete(zlib_out_3.data());
    //delete(zlib_out_4.data());
    // Close file
    //inFile.close();
    //outFile.close();
    return enbytes;
}

//int validate(std::string& inFile_name, std::string& outFile_name) {
//    std::string command = "cmp " + inFile_name + " " + outFile_name;
//    int ret = system(command.c_str());
//    return ret;
//}

// Constructor
xil_zlib::xil_zlib(const std::string& binaryFileName, uint8_t flow, uint32_t compress_block_size, uint64_t input_size) {
    // Zlib Compression Binary Name
    init(binaryFileName, flow);

    // printf("C_COMPUTE_UNIT \n");
    uint32_t block_size_in_kb = compress_block_size;
    //uint32_t block_size_in_bytes = block_size_in_kb * 1024;
    uint32_t overlap_buf_count = OVERLAP_BUF_COUNT;
    uint32_t host_buffer_size = PARALLEL_ENGINES * block_size_in_kb * 1024;
    uint32_t HOST_BUFFER_SIZE = host_buffer_size;
    uint32_t MAX_NUMBER_BLOCKS =  (HOST_BUFFER_SIZE / (32 * 1024));
    host_buffer_size = ((host_buffer_size - 1) / block_size_in_kb + 1) * block_size_in_kb;
    uint32_t t;
    //uint32_t chunk_num;
    //chunk_num = (input_size - 1)/HOST_BUFFER_SIZE + 1;
    //chunk_num = (chunk_num - 1)/8 + 1;
    //if(input_size > 12*1024*1024)
    //    chunk_num = 0;
    for (int i = 0; i < MAX_CCOMP_UNITS; i++) {
        for (int j = 0; j < OVERLAP_BUF_COUNT; j++) {
            // Index calculation
            h_buf_in_s[i][j].resize(PARALLEL_ENGINES * HOST_BUFFER_SIZE);
            //h_buf_out[i][j].resize(PARALLEL_ENGINES * HOST_BUFFER_SIZE * 4);
            h_buf_zlibout[i][j].resize(PARALLEL_ENGINES * HOST_BUFFER_SIZE * 33/32);
            h_blksize[i][j].resize(MAX_NUMBER_BLOCKS);
            h_compressSize[i][j].resize(MAX_NUMBER_BLOCKS);
            //h_compressSize[i][j].resize(256);
            h_dyn_ltree_freq[i][j].resize(PARALLEL_ENGINES * LTREE_SIZE);
            h_dyn_dtree_freq[i][j].resize(PARALLEL_ENGINES * DTREE_SIZE);
            h_dyn_bltree_freq[i][j].resize(PARALLEL_ENGINES * BLTREE_SIZE);
            h_dyn_ltree_codes[i][j].resize(PARALLEL_ENGINES * LTREE_SIZE);
            h_dyn_dtree_codes[i][j].resize(PARALLEL_ENGINES * DTREE_SIZE);
            h_dyn_bltree_codes[i][j].resize(PARALLEL_ENGINES * BLTREE_SIZE);
            h_dyn_ltree_blen[i][j].resize(PARALLEL_ENGINES * LTREE_SIZE);
            h_dyn_dtree_blen[i][j].resize(PARALLEL_ENGINES * DTREE_SIZE);
            h_dyn_bltree_blen[i][j].resize(PARALLEL_ENGINES * BLTREE_SIZE);

            h_buff_max_codes[i][j].resize(PARALLEL_ENGINES * MAXCODE_SIZE);
        }
    }
    //for(t = 0; t < MAX_CHUNK; t++){
    //for(t = 0; t < chunk_num; t++){
    //    for (int i = 0; i < MAX_CCOMP_UNITS; i++) {
    //        for (int j = 0; j < OVERLAP_BUF_COUNT; j++) {
    //            h_buf_in[t][i][j].resize(PARALLEL_ENGINES * HOST_BUFFER_SIZE);
    //        }
    //    }
    //}
    //ext mem
    //for(t = 0; t < MAX_CHUNK; t++) {
    //for(t = 0; t < chunk_num; t++) {
    //    for (uint32_t cu = 0; cu < C_COMPUTE_UNIT; cu++) {
    //        for (uint32_t flag = 0; flag < overlap_buf_count; flag++) {
    //            ext_buffer_input[t][cu][flag].obj = h_buf_in[t][cu][flag].data();
    //            ext_buffer_input[t][cu][flag].param = 0 ;
    //        if(cu == 0) {
    //            ext_buffer_input[t][cu][flag].flags = bank[cu*4];
    //        } else if(cu == 1) {
    //            ext_buffer_input[t][cu][flag].flags = bank[cu*4];
    //        } else if(cu == 2) {
    //            ext_buffer_input[t][cu][flag].flags = bank[cu*4];
    //        } else if(cu == 3) {
    //            ext_buffer_input[t][cu][flag].flags = bank[cu*4];
    //        }
    //        
    //        }
    //    }
    //}
    for (uint32_t cu = 0; cu < C_COMPUTE_UNIT; cu++) {
        for (uint32_t flag = 0; flag < overlap_buf_count; flag++) {
            //ext_buffer_input_s[cu][flag].obj = h_buf_in_s[cu][flag].data();
            //ext_buffer_input_s[cu][flag].param = 0 ;
            ext_buffer_inblk_size[cu][flag].obj = h_blksize[cu][flag].data(); // Setting Obj and Param to Zero
            ext_buffer_inblk_size[cu][flag].param = 0;
            ext_buffer_zlib_output[cu][flag].obj = h_buf_zlibout[cu][flag].data(); // Setting Obj and Param to Zero
            ext_buffer_zlib_output[cu][flag].param = 0;
            ext_buffer_compress_size[cu][flag].obj = h_compressSize[cu][flag].data(); // Setting Obj and Param to Zero
            ext_buffer_compress_size[cu][flag].param = 0;
            #if PLATFORM == 250
            if(cu == 0) {
                //ext_buffer_input_s[cu][flag].flags =       bank[4*cu + 0];
                ext_buffer_inblk_size[cu][flag].flags =    XCL_MEM_DDR_BANK0;
                ext_buffer_zlib_output[cu][flag].flags =   XCL_MEM_DDR_BANK0;
                ext_buffer_compress_size[cu][flag].flags = XCL_MEM_DDR_BANK0;
            } else if(cu == 1) {
                //ext_buffer_input_s[cu][flag].flags =        bank[4*cu + 0];
                ext_buffer_inblk_size[cu][flag].flags =     XCL_MEM_DDR_BANK1;
                ext_buffer_zlib_output[cu][flag].flags =    XCL_MEM_DDR_BANK1;
                ext_buffer_compress_size[cu][flag].flags =  XCL_MEM_DDR_BANK1;
            } else if(cu == 2) {
                //ext_buffer_input_s[cu][flag].flags =       bank[4*cu + 0];
                ext_buffer_inblk_size[cu][flag].flags =    XCL_MEM_DDR_BANK2;
                ext_buffer_zlib_output[cu][flag].flags =   XCL_MEM_DDR_BANK2;
                ext_buffer_compress_size[cu][flag].flags = XCL_MEM_DDR_BANK2;
            } else if(cu == 3) {
                //ext_buffer_input_s[cu][flag].flags =       bank[4*cu + 0];
                ext_buffer_inblk_size[cu][flag].flags =    XCL_MEM_DDR_BANK3;
                ext_buffer_zlib_output[cu][flag].flags =   XCL_MEM_DDR_BANK3;
                ext_buffer_compress_size[cu][flag].flags = XCL_MEM_DDR_BANK3;
            }
            #else
            if(cu == 0) {
                //ext_buffer_input_s[cu][flag].flags =       bank[4*cu + 0];
                ext_buffer_inblk_size[cu][flag].flags =    bank[4*cu + 1];
                ext_buffer_zlib_output[cu][flag].flags =   bank[4*cu + 2];
                ext_buffer_compress_size[cu][flag].flags = bank[4*cu + 3];
            } else if(cu == 1) {
                //ext_buffer_input_s[cu][flag].flags =        bank[4*cu + 0];
                ext_buffer_inblk_size[cu][flag].flags =     bank[4*cu + 1];
                ext_buffer_zlib_output[cu][flag].flags =    bank[4*cu + 2];
                ext_buffer_compress_size[cu][flag].flags =  bank[4*cu + 3];
            } else if(cu == 2) {
                //ext_buffer_input_s[cu][flag].flags =       bank[4*cu + 0];
                ext_buffer_inblk_size[cu][flag].flags =    bank[4*cu + 1];
                ext_buffer_zlib_output[cu][flag].flags =   bank[4*cu + 2];
                ext_buffer_compress_size[cu][flag].flags = bank[4*cu + 3];
            } else if(cu == 3) {
                //ext_buffer_input_s[cu][flag].flags =       bank[4*cu + 0];
                ext_buffer_inblk_size[cu][flag].flags =    bank[4*cu + 1];
                ext_buffer_zlib_output[cu][flag].flags =   bank[4*cu + 2];
                ext_buffer_compress_size[cu][flag].flags = bank[4*cu + 3];
            }

            #endif
        }
    }
    // Device buffer allocation
    
    //for(t = 0; t < MAX_CHUNK; t++){
    //for(t = 0; t < chunk_num; t++) {
    //    for (uint32_t cu = 0; cu < C_COMPUTE_UNIT; cu++) {
    //        for (uint32_t flag = 0; flag < overlap_buf_count; flag++) {
    //            buffer_input[t][cu][flag] = new cl::Buffer(*m_context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
    //                                                host_buffer_size, &ext_buffer_input[t][cu][flag]);
    //        
    //        }
    //    }
    //}
    for (uint32_t cu = 0; cu < C_COMPUTE_UNIT; cu++) {
        for (uint32_t flag = 0; flag < overlap_buf_count; flag++) {

            //buffer_input_s[cu][flag] = new cl::Buffer(*m_context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            //                                        host_buffer_size, &ext_buffer_input_s[cu][flag]);

            buffer_inblk_size[cu][flag] = new cl::Buffer(*m_context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                         384 * sizeof(uint32_t), &ext_buffer_inblk_size[cu][flag]);

            buffer_zlib_output[cu][flag] = new cl::Buffer(*m_context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                                          host_buffer_size *33/32, &ext_buffer_zlib_output[cu][flag]);
            //buffer_zlib_output[cu][flag] = new cl::Buffer(*m_context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
            //                                              host_buffer_size, &ext_buffer_zlib_output[cu][flag]);

            buffer_compress_size[cu][flag] =
                new cl::Buffer(*m_context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, 384 * sizeof(uint32_t),
                               &ext_buffer_compress_size[cu][flag]);

        }
    }
    
    //for (int i = 0; i < MAX_DDCOMP_UNITS; i++) {
    //    h_dbuf_in[i].resize(PARALLEL_ENGINES * HOST_BUFFER_SIZE);
    //    h_dbuf_zlibout[i].resize(PARALLEL_ENGINES * HOST_BUFFER_SIZE * 10);
    //    h_dcompressSize[i].resize(MAX_NUMBER_BLOCKS);
    //}
}

// Destructor
xil_zlib::~xil_zlib() {
    release();
    uint32_t overlap_buf_count = OVERLAP_BUF_COUNT;
    for (uint32_t t = 0; t < MAX_CHUNK; t++) {
        for (uint32_t cu = 0; cu < C_COMPUTE_UNIT; cu++) {
            for (uint32_t flag = 0; flag < overlap_buf_count; flag++) {
                delete (buffer_input[t][cu][flag]);
            }
        }
    }
    for (uint32_t cu = 0; cu < C_COMPUTE_UNIT; cu++) {
        for (uint32_t flag = 0; flag < overlap_buf_count; flag++) {
            delete (buffer_input_s[cu][flag]);
            //delete (buffer_lz77_output[cu][flag]);
            delete (buffer_zlib_output[cu][flag]);
            delete (buffer_compress_size[cu][flag]);
            delete (buffer_inblk_size[cu][flag]);

            delete (buffer_dyn_ltree_freq[cu][flag]);
            delete (buffer_dyn_dtree_freq[cu][flag]);
            delete (buffer_dyn_bltree_freq[cu][flag]);

            delete (buffer_dyn_ltree_codes[cu][flag]);
            delete (buffer_dyn_dtree_codes[cu][flag]);
            delete (buffer_dyn_bltree_codes[cu][flag]);

            delete (buffer_dyn_ltree_blen[cu][flag]);
            delete (buffer_dyn_dtree_blen[cu][flag]);
            delete (buffer_dyn_bltree_blen[cu][flag]);
        }
    }
}

int xil_zlib::init(const std::string& binaryFileName, uint8_t flow) {
    // The get_xil_devices will return vector of Xilinx Devices
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Creating Context and Command Queue for selected Device
    m_context = new cl::Context(device);
    // m_q = new cl::CommandQueue(*m_context, device, CL_QUEUE_PROFILING_ENABLE);

    for (uint8_t i = 0; i < C_COMPUTE_UNIT * OVERLAP_BUF_COUNT; i++) {
        m_q[i] = new cl::CommandQueue(*m_context, device, CL_QUEUE_PROFILING_ENABLE);
    }

    //for (uint8_t flag = 0; flag < D_COMPUTE_UNIT; flag++) {
    //    m_q_dec[flag] = new cl::CommandQueue(*m_context, device, CL_QUEUE_PROFILING_ENABLE);
    //}
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Found Device=" << device_name.c_str() << std::endl;

    // import_binary() command will find the OpenCL binary file created using the
    // xocc compiler load into OpenCL Binary and return as Binaries
    // OpenCL and it can contain many functions which can be executed on the
    // device.
    // std::string binaryFile = xcl::find_binary_file(device_name,binaryFileName.c_str());
    // cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    auto fileBuf = xcl::read_binary_file(binaryFileName);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    devices.resize(1);
    m_program = new cl::Program(*m_context, devices, bins);

    if (flow == 0 || flow == 2) {
        //// Create Tree generation kernel
        for (int i = 0; i < T_COMPUTE_UNIT; i++) {
            zlib_kernel[i] = new cl::Kernel(*m_program,zlib_kernel_names[i].c_str());
        }
    }
    //if (flow == 1 || flow == 2) {
    //    // Create Decompress kernel
    //    for (int i = 0; i < D_COMPUTE_UNIT; i++) {
    //        decompress_kernel[i] = new cl::Kernel(*m_program, decompress_kernel_names[i].c_str());
    //    }
    //}

    return 0;
}

int xil_zlib::release() {
    //if (!m_bin_flow) {
        //for (int i = 0; i < C_COMPUTE_UNIT; i++) delete (compress_kernel[i]);
        //for (int i = 0; i < H_COMPUTE_UNIT; i++) delete (huffman_kernel[i]);
        //for (int i = 0; i < T_COMPUTE_UNIT; i++) delete (treegen_kernel[i]);
    for (int i = 0; i < Z_COMPUTE_UNIT; i++) delete (zlib_kernel[i]);
    //} else if (m_bin_flow) {
    //    for (int i = 0; i < D_COMPUTE_UNIT; i++) delete (decompress_kernel[i]);
    //}

    delete (m_program);

    for (uint8_t i = 0; i < C_COMPUTE_UNIT * OVERLAP_BUF_COUNT; i++) {
        delete (m_q[i]);
    }

    //for (uint8_t flag = 0; flag < D_COMPUTE_UNIT; flag++) {
    //    delete (m_q_dec[flag]);
    //}

    delete (m_context);

    return 0;
}
#define MAX_DATA_PROCESS_SIZE 800*1024*1024
uint32_t xil_zlib::compress(uint8_t* in, uint8_t* out1,uint8_t* out2,uint8_t* out3,uint8_t* out4, uint32_t compress_block_size,
                            uint32_t data_block_num,
                            uint32_t * data_block_size_in,
                            uint32_t * compress_size_out_1,
                            uint32_t * compress_size_out_2,
                            uint32_t * compress_size_out_3,
                            uint32_t * compress_size_out_4
                            ) {
    //////printme("In compress \n");
    uint32_t block_size_in_kb = compress_block_size;
    uint32_t block_size_in_bytes = block_size_in_kb * 1024;
    uint32_t overlap_buf_count = OVERLAP_BUF_COUNT;
    uint32_t host_buffer_size = PARALLEL_ENGINES * compress_block_size * 1024;
    uint32_t input_size = data_block_num * 32768;

    uint32_t total_chunks = (input_size - 1) / host_buffer_size + 1;
    //printf("total chunks %d \n", total_chunks);
    uint32_t pe_input_offset[4];
    uint32_t total_loop_cnt = (input_size - 1)/(MAX_DATA_PROCESS_SIZE) + 1;
    uint32_t pe_total_chunk_num[4];
    if(total_chunks <= 4){
        for(int i = 0; i < 4; i++){
            if(i < total_chunks){
                pe_total_chunk_num[i] = 1;
                }
            else{
                pe_total_chunk_num[i] = 0;
                }
            }
        }
    else{
        int average_chunks_pe = total_chunks / 4;
        int remainder_chunks = total_chunks % 4;
        for(int i = 0; i < 4; i++){
            if(i < remainder_chunks){
                pe_total_chunk_num[i] = average_chunks_pe + 1;
                }
            else{
                pe_total_chunk_num[i] = average_chunks_pe;
                }
            } 
    }
    pe_input_offset[0] = 0;
    pe_input_offset[1] = pe_total_chunk_num[0] * host_buffer_size;
    pe_input_offset[2] = (pe_total_chunk_num[0] + pe_total_chunk_num[1]) * host_buffer_size;
    pe_input_offset[3] = (pe_total_chunk_num[0] + pe_total_chunk_num[1] + pe_total_chunk_num[2]) * host_buffer_size;

    if (total_chunks < 2) overlap_buf_count = 1;

    // Find out the size of each chunk spanning entire file
    uint32_t sizeOfChunk[total_chunks];
    uint32_t blocksPerChunk[total_chunks];
    uint32_t idx = 0;
    //uint32_t block_size_idx = 0;
    uint32_t block_size_idx[4] = {0,0,0,0};
    uint32_t pe_data_bk_size_offset[4] = {0,0,0,0};
    pe_data_bk_size_offset[0] = 0;
    pe_data_bk_size_offset[1] = pe_input_offset[1]/32768;
    pe_data_bk_size_offset[2] = pe_input_offset[2]/32768;
    pe_data_bk_size_offset[3] = pe_input_offset[3]/32768;
    uint32_t chunk_num_pe[4];
    uint32_t data_size_pe[4];
    uint32_t outIdx_1 = 0;
    uint32_t outIdx_2 = 0;
    uint32_t outIdx_3 = 0;
    uint32_t outIdx_4 = 0;
    uint32_t compress_size_index_1 = 0;
    uint32_t compress_size_index_2 = 0;
    uint32_t compress_size_index_3 = 0;
    uint32_t compress_size_index_4 = 0;
    uint32_t current_total_chunks;
    for(uint32_t data_cnt = 0; data_cnt < total_loop_cnt; data_cnt++){
        cl_buffer_region buffer_in_sub_info[128];
        cl::Buffer buffer_in_sub[128];
        cl::Buffer* buffer_total_in[4];
        //printf("data cnt %d \n",data_cnt);
        uint32_t current_input_size;
        if(input_size >= MAX_DATA_PROCESS_SIZE){
            current_input_size = MAX_DATA_PROCESS_SIZE;
            input_size = input_size - MAX_DATA_PROCESS_SIZE;
            }
        else{
            current_input_size = input_size;
            }
        current_total_chunks = (current_input_size - 1)/host_buffer_size + 1;
        if(current_total_chunks <= 4){
            for(int i = 0; i < 4; i++){
                if(i < current_total_chunks){
                    chunk_num_pe[i] = 1;
                    }
                else{
                    chunk_num_pe[i] = 0;
                    }
                }
            }
        else{
            int average_chunks_pe = current_total_chunks / 4;
            int remainder_chunks = current_total_chunks % 4;
            for(int i = 0; i < 4; i++){
                if(i < remainder_chunks){
                    chunk_num_pe[i] = average_chunks_pe + 1;
                    }
                else{
                    chunk_num_pe[i] = average_chunks_pe;
                    }
                } 
        }
        data_size_pe[0] = chunk_num_pe[0] * host_buffer_size;
        data_size_pe[1] = chunk_num_pe[1] * host_buffer_size;
        data_size_pe[2] = chunk_num_pe[2] * host_buffer_size;
        data_size_pe[3] = current_input_size - data_size_pe[0] - data_size_pe[1] - data_size_pe[2];

        //printf("input size %d \n", input_size);
        //printf("quartor input size %d \n", input_size/4);
        #if PLATFORM == 250
        cl_mem_ext_ptr_t buffer_words_ext[4];
        buffer_words_ext[0].flags = XCL_MEM_DDR_BANK0;
        buffer_words_ext[0].param = 0;
        buffer_words_ext[0].obj   = &in[pe_input_offset[0] + data_cnt*MAX_DATA_PROCESS_SIZE/4];
        buffer_words_ext[1].flags = XCL_MEM_DDR_BANK1;
        buffer_words_ext[1].param = 0;
        buffer_words_ext[1].obj   = &in[pe_input_offset[1] + data_cnt*MAX_DATA_PROCESS_SIZE/4];
        buffer_words_ext[2].flags = XCL_MEM_DDR_BANK2;
        buffer_words_ext[2].param = 0;
        buffer_words_ext[2].obj   = &in[pe_input_offset[2] + data_cnt*MAX_DATA_PROCESS_SIZE/4];
        buffer_words_ext[3].flags = XCL_MEM_DDR_BANK3;
        buffer_words_ext[3].param = 0;
        buffer_words_ext[3].obj   = &in[pe_input_offset[3] + data_cnt*MAX_DATA_PROCESS_SIZE/4];
        #else
        cl_mem_ext_ptr_t buffer_words_ext[4];
        buffer_words_ext[0].flags = bank[0];
        buffer_words_ext[0].param = 0;
        buffer_words_ext[0].obj   = &in[pe_input_offset[0] + data_cnt*MAX_DATA_PROCESS_SIZE/4];
        buffer_words_ext[1].flags = bank[4];
        buffer_words_ext[1].param = 0;
        buffer_words_ext[1].obj   = &in[pe_input_offset[1] + data_cnt*MAX_DATA_PROCESS_SIZE/4];
        buffer_words_ext[2].flags = bank[8];
        buffer_words_ext[2].param = 0;
        buffer_words_ext[2].obj   = &in[pe_input_offset[2] + data_cnt*MAX_DATA_PROCESS_SIZE/4];
        buffer_words_ext[3].flags = bank[12];
        buffer_words_ext[3].param = 0;
        buffer_words_ext[3].obj   = &in[pe_input_offset[3] + data_cnt*MAX_DATA_PROCESS_SIZE/4];
        #endif
        buffer_total_in[0] =new cl::Buffer(*m_context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (data_size_pe[0])*sizeof(uint8_t),&buffer_words_ext[0]);
        buffer_total_in[1] =new cl::Buffer(*m_context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (data_size_pe[1])*sizeof(uint8_t),&buffer_words_ext[1]);
        buffer_total_in[2] =new cl::Buffer(*m_context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (data_size_pe[2])*sizeof(uint8_t),&buffer_words_ext[2]);
        buffer_total_in[3] =new cl::Buffer(*m_context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (data_size_pe[3])*sizeof(uint8_t),&buffer_words_ext[3]);
        //else{
        //cl_mem_ext_ptr_t buffer_words_ext[4];
        //buffer_words_ext[0].flags = bank[0];
        //buffer_words_ext[0].param = 0;
        //buffer_words_ext[0].obj   = &in[pe_input_offset[0] + data_cnt*MAX_DATA_PROCESS_SIZE/4];
        //buffer_words_ext[1].flags = bank[4];
        //buffer_words_ext[1].param = 0;
        //buffer_words_ext[1].obj   = &in[pe_input_offset[1] + data_cnt*MAX_DATA_PROCESS_SIZE/4];
        //buffer_words_ext[2].flags = bank[8];
        //buffer_words_ext[2].param = 0;
        //buffer_words_ext[2].obj   = &in[pe_input_offset[2] + data_cnt*MAX_DATA_PROCESS_SIZE/4];
        //buffer_words_ext[3].flags = bank[12];
        //buffer_words_ext[3].param = 0;
        //buffer_words_ext[3].obj   = &in[pe_input_offset[3] + data_cnt*MAX_DATA_PROCESS_SIZE/4];
        //buffer_total_in[1] =new cl::Buffer(*m_context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (data_size_pe[0])*sizeof(uint8_t),&buffer_words_ext[0]);
        //buffer_total_in[2] =new cl::Buffer(*m_context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (data_size_pe[1])*sizeof(uint8_t),&buffer_words_ext[1]);
        //buffer_total_in[3] =new cl::Buffer(*m_context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (data_size_pe[2])*sizeof(uint8_t),&buffer_words_ext[2]);
        //buffer_total_in[4] =new cl::Buffer(*m_context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (data_size_pe[3])*sizeof(uint8_t),&buffer_words_ext[3]);
        //}
        //m_q[0]->enqueueMigrateMemObjects({*buffer_total_in[0], *buffer_total_in[1],*buffer_total_in[2],*buffer_total_in[3]}, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED);
        //cl::Buffer buffer_total_in(*m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, input_size*sizeof(uint8_t),in);
        //cl_buffer_region buffer_in_sub_info[current_total_chunks];
        //cl::Buffer buffer_in_sub[current_total_chunks];
        //printf("step 1 \n");
        for (uint32_t i = 0; i < current_input_size; i += host_buffer_size, idx++) {
            uint32_t chunk_size = host_buffer_size;
            if (chunk_size + i > current_input_size) {
                chunk_size = current_input_size - i;
            }
            // Update size of each chunk buffer
            sizeOfChunk[idx] = chunk_size;
            // Calculate sub blocks of size BLOCK_SIZE_IN_KB for each chunk
            // 2MB(example)
            // Figure out blocks per chunk
            uint32_t nblocks = (chunk_size - 1) / block_size_in_bytes + 1;
            blocksPerChunk[idx] = nblocks;
        }
        //printf("step 2 \n");
        //printf("total chunks %d \n", total_chunks);
        uint32_t pe_sizeOfChunk[4][128];
        uint32_t pe_blocksPerChunk[4][128];
        idx = 0;
        for(uint32_t i = 0; i < data_size_pe[0]; i+=host_buffer_size,idx++) {
            uint32_t chunk_size = host_buffer_size;
            if (chunk_size + i > data_size_pe[0]) {
                chunk_size = data_size_pe[0] - i;
            }
            pe_sizeOfChunk[0][idx] = chunk_size;
            uint32_t nblocks = (chunk_size - 1) / block_size_in_bytes + 1;
            pe_blocksPerChunk[0][idx] = nblocks;
        }       
        idx = 0;
        for(uint32_t i = 0; i < data_size_pe[1]; i+=host_buffer_size,idx++) {
            uint32_t chunk_size = host_buffer_size;
            if (chunk_size + i > data_size_pe[1]) {
                chunk_size = data_size_pe[1] - i;
            }
            pe_sizeOfChunk[1][idx] = chunk_size;
            uint32_t nblocks = (chunk_size - 1) / block_size_in_bytes + 1;
            pe_blocksPerChunk[1][idx] = nblocks;
        }       
        idx = 0;
        for(uint32_t i = 0; i < data_size_pe[2]; i+=host_buffer_size,idx++) {
            uint32_t chunk_size = host_buffer_size;
            if (chunk_size + i > data_size_pe[2]) {
                chunk_size = data_size_pe[2] - i;
            }
            pe_sizeOfChunk[2][idx] = chunk_size;
            uint32_t nblocks = (chunk_size - 1) / block_size_in_bytes + 1;
            pe_blocksPerChunk[2][idx] = nblocks;
        }       
        idx = 0;
        for(uint32_t i = 0; i < data_size_pe[3]; i+=host_buffer_size,idx++) {
            uint32_t chunk_size = host_buffer_size;
            if (chunk_size + i > data_size_pe[3]) {
                chunk_size = data_size_pe[3] - i;
            }
            pe_sizeOfChunk[3][idx] = chunk_size;
            uint32_t nblocks = (chunk_size - 1) / block_size_in_bytes + 1;
            pe_blocksPerChunk[3][idx] = nblocks;
        }       
        //for(uint32_t i = 0; i < total_chunks; i++){
        //    //printf("create buffer %d \n",i);
        //    buffer_in_sub_info[i] = {(i/4)*host_buffer_size*sizeof(uint8_t), sizeOfChunk[i]*sizeof(uint8_t)};
        //    buffer_in_sub[i] = buffer_total_in[i%4].createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &buffer_in_sub_info[i]);
        //    }
        //printf("step 1 \n");
        for(uint32_t i = 0; i < chunk_num_pe[0]; i++){
            buffer_in_sub_info[i] = {i*host_buffer_size*sizeof(uint8_t), pe_sizeOfChunk[0][i]*sizeof(uint8_t)};
            buffer_in_sub[i*4 + 0] = buffer_total_in[0]->createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &buffer_in_sub_info[i]);
            //printf("buffer index %d \n", i*4);
            //printf("offset %d \n", i*host_buffer_size);
            //printf("size %d \n", pe_sizeOfChunk[0][i]);
            }
        //printf("step 2 \n");
        for(uint32_t i = 0; i < chunk_num_pe[1]; i++){
            buffer_in_sub_info[i] = {i*host_buffer_size*sizeof(uint8_t), pe_sizeOfChunk[1][i]*sizeof(uint8_t)};
            buffer_in_sub[i*4 + 1] = buffer_total_in[1]->createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &buffer_in_sub_info[i]);
            //printf("buffer index %d \n", i*4);
            //printf("offset %d \n", i*host_buffer_size);
            //printf("size %d \n", pe_sizeOfChunk[1][i]);
            }
        //printf("step 3 \n");
        for(uint32_t i = 0; i < chunk_num_pe[2]; i++){
            buffer_in_sub_info[i] = {i*host_buffer_size*sizeof(uint8_t), pe_sizeOfChunk[2][i]*sizeof(uint8_t)};
            buffer_in_sub[i*4 + 2] = buffer_total_in[2]->createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &buffer_in_sub_info[i]);
            //printf("buffer index %d \n", i*4);
            //printf("offset %d \n", i*host_buffer_size);
            //printf("size %d \n", pe_sizeOfChunk[2][i]);
            }
        //printf("step 4 \n");
        for(uint32_t i = 0; i < chunk_num_pe[3]; i++){
            buffer_in_sub_info[i] = {i*host_buffer_size*sizeof(uint8_t), pe_sizeOfChunk[3][i]*sizeof(uint8_t)};
            buffer_in_sub[i*4 + 3] = buffer_total_in[3]->createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &buffer_in_sub_info[i]);
            //printf("buffer index %d \n", i*4);
            //printf("offset %d \n", i*host_buffer_size);
            //printf("size %d \n", pe_sizeOfChunk[3][i]);
            }
        //printf("step 5 \n");
        //printf("total chunk %d \n", total_chunks);
        //printf("pe 0 chunk %d \n", chunk_num_pe[0]);
        //printf("pe 1 chunk %d \n", chunk_num_pe[1]);
        //printf("pe 2 chunk %d \n", chunk_num_pe[2]);
        //printf("pe 3 chunk %d \n", chunk_num_pe[3]);
        //printf("pe 0 data size %d \n",data_size_pe[0]);
        //printf("pe 1 data size %d \n",data_size_pe[1]);
        //printf("pe 2 data size %d \n",data_size_pe[2]);
        //printf("pe 3 data size %d \n",data_size_pe[3]);
        // Counter which helps in tracking
        // Output buffer index

        // Track the lags of respective chunks for left over handling
        int chunk_flags[current_total_chunks];
        int cu_order[current_total_chunks];

        // Finished bricks
        int completed_bricks = 0;

        int flag = 0;
        uint32_t lcl_cu = 0;

        uint8_t cunits = (uint8_t)C_COMPUTE_UNIT;
        uint8_t queue_idx = 0;
        //std::thread memin_thread[C_COMPUTE_UNIT];
        //auto kernel_start_0 = std::chrono::high_resolution_clock::now();
        //printf("test start 0 %u \n",kernel_start_0);
overlap:
    for (uint32_t brick = 0, itr = 0; brick < current_total_chunks; /*brick += C_COMPUTE_UNIT,*/ itr++, flag = !flag) {
        if (cunits > 1)
            queue_idx = flag * OVERLAP_BUF_COUNT * 2;
        else
            queue_idx = flag;

        if (current_total_chunks > 2)
            lcl_cu = C_COMPUTE_UNIT;
        else
            lcl_cu = 1;

        if (brick + lcl_cu > current_total_chunks) lcl_cu = current_total_chunks - brick;
        {
            for (uint32_t cu = 0; cu < lcl_cu; cu++) {
                chunk_flags[brick + cu] = flag;
                cu_order[brick + cu] = cu;

                // Wait for read events
                if (itr >= 2) {
                    // Wait on current flag previous operation to finish
                    
                    m_q[queue_idx + cu]->finish();

                    // Completed bricks counter
                    completed_bricks++;
                    //sched_param sch;
                    uint32_t index = 0;
                    //uint32_t brick_flag_idx = brick - (C_COMPUTE_UNIT * overlap_buf_count - cu);
                    uint32_t brick_flag_idx = (brick - (C_COMPUTE_UNIT * overlap_buf_count))/4;
                    for (uint32_t bIdx = 0; bIdx < pe_blocksPerChunk[cu][brick_flag_idx]; bIdx++, index += block_size_in_bytes) {
                        uint32_t block_size = block_size_in_bytes;
                        if (index + block_size > pe_sizeOfChunk[cu][brick_flag_idx]) {
                            block_size = pe_sizeOfChunk[cu][brick_flag_idx] - index;
                        }
                        uint32_t no_sub_block = (block_size - 1) / 32768 + 1;
                        for(uint32_t sub_idx = 0; sub_idx < no_sub_block; sub_idx++){
                            uint32_t compressed_size = (h_compressSize[cu][flag].data())[sub_idx + bIdx * compress_block_size/32];
                            //printf("read size from cu %d \n",cu);
                            //printf("compressed_size %d \n",compressed_size);
                            if(cu == 0){
                                std::memcpy(&out1[outIdx_1], &h_buf_zlibout[cu][flag].data()[bIdx* block_size_in_bytes*33/32 + sub_idx*32768*33/32],
                                        compressed_size);
                                outIdx_1 += compressed_size;
                                compress_size_out_1[compress_size_index_1++]= compressed_size;
                            }
                            else if(cu == 1){
                                std::memcpy(&out2[outIdx_2], &h_buf_zlibout[cu][flag].data()[bIdx* block_size_in_bytes*33/32 + sub_idx*32768*33/32],
                                        compressed_size);
                                outIdx_2 += compressed_size;
                                compress_size_out_2[compress_size_index_2++]= compressed_size;
                                }
                            else if(cu == 2){
                                std::memcpy(&out3[outIdx_3], &h_buf_zlibout[cu][flag].data()[bIdx* block_size_in_bytes*33/32 + sub_idx*32768*33/32],
                                        compressed_size);
                                outIdx_3 += compressed_size;
                                compress_size_out_3[compress_size_index_3++]= compressed_size;
                                }
                            else if(cu == 3){
                                std::memcpy(&out4[outIdx_4], &h_buf_zlibout[cu][flag].data()[bIdx* block_size_in_bytes*33/32 + sub_idx*32768*33/32],
                                        compressed_size);
                                outIdx_4 += compressed_size;
                                compress_size_out_4[compress_size_index_4++]= compressed_size;
                                }
                        }
                    }
                } // If condition which reads huffman output for 0 or 1 location

                // Figure out block sizes per brick
                uint32_t idxblk = 0;
                for (uint32_t i = 0; i < pe_sizeOfChunk[cu][brick/4]; i += 32768) {
                    uint32_t block_size = 32768;
                    block_size = data_block_size_in[block_size_idx[cu] + pe_data_bk_size_offset[cu]]; 
                    block_size_idx[cu]++;
                    //if(block_size != 32768){
                        //printf("block_size %d \n",block_size);
                        //printf("cu %d \n",cu);
                        //printf("block_size_idx %d \n",block_size_idx[cu]);
                        //}
                    (h_blksize[cu][flag]).data()[idxblk++] = block_size;
                
                }
                //if(itr < 2){
                //    std::memcpy(h_buf_in_s[cu][flag].data(), &in[(brick + cu) * host_buffer_size], sizeOfChunk[brick + cu]);
                //}
                // Set kernel arguments
                int narg = 0;
                //printf("brick + cu %d \n", brick+cu);
                //printf("input size %d \n",pe_sizeOfChunk[cu][brick/4]);
                //printf("set kernel argument \n");
                (zlib_kernel[cu])->setArg(narg++, buffer_in_sub[brick+cu]);
                (zlib_kernel[cu])->setArg(narg++, *(buffer_inblk_size[cu][flag]));
                (zlib_kernel[cu])->setArg(narg++, block_size_in_kb);
                //(zlib_kernel[cu])->setArg(narg++, sizeOfChunk[brick + cu]);
                (zlib_kernel[cu])->setArg(narg++, pe_sizeOfChunk[cu][brick/4]);
                (zlib_kernel[cu])->setArg(narg++, blocksPerChunk[brick]);
                (zlib_kernel[cu])->setArg(narg++, *(buffer_zlib_output[cu][flag]));
                (zlib_kernel[cu])->setArg(narg++, *(buffer_compress_size[cu][flag]));
                m_q[queue_idx + cu]->enqueueMigrateMemObjects({(buffer_in_sub[brick+cu]), *(buffer_inblk_size[cu][flag])},
                                                              0 /* 0 means from host*/);
                m_q[queue_idx + cu]->enqueueTask(*zlib_kernel[cu]);
                m_q[queue_idx + cu]->enqueueMigrateMemObjects(
                    {*(buffer_zlib_output[cu][flag]), *(buffer_compress_size[cu][flag])}, CL_MIGRATE_MEM_OBJECT_HOST);
                //printf("start kernel \n");
            } // Internal loop runs on compute units
        }

        if (current_total_chunks > 2)
            brick += C_COMPUTE_UNIT;
        else
            brick++;

    } // Main overlap loop
    //auto kernel_start_3 = std::chrono::high_resolution_clock::now();
    //printf("test start 3 %u \n",kernel_start_3);

    for (uint8_t i = 0; i < C_COMPUTE_UNIT * OVERLAP_BUF_COUNT; i++) {
        m_q[i]->flush();
        m_q[i]->finish();
    }
    //auto kernel_start_4 = std::chrono::high_resolution_clock::now();
    //printf("test start 4 %u \n",kernel_start_4);

    uint32_t leftover = current_total_chunks - completed_bricks;
    uint32_t stride = 0;

    if ((current_total_chunks < overlap_buf_count * C_COMPUTE_UNIT))
        stride = overlap_buf_count * C_COMPUTE_UNIT;
    else
        stride = current_total_chunks;

    // Handle leftover bricks
    for (uint32_t ovr_itr = 0, brick = stride - overlap_buf_count * C_COMPUTE_UNIT; ovr_itr < leftover;
         ovr_itr += C_COMPUTE_UNIT, brick += C_COMPUTE_UNIT) {
        lcl_cu = C_COMPUTE_UNIT;
        if (ovr_itr + lcl_cu > leftover) lcl_cu = leftover - ovr_itr;
        //printf("lcl cu %d \n", lcl_cu);
        // Handle multiple bricks with multiple CUs
        for (uint32_t j = 0; j < lcl_cu; j++) {
            int cu = cu_order[brick + j];
            int flag = chunk_flags[brick + j];
            // Run over each block within brick
            uint32_t index = 0;
            //uint32_t brick_flag_idx = brick + j;
            uint32_t brick_flag_idx = brick/4;
            // Copy the data from various blocks in concatinated manner
            for (uint32_t bIdx = 0; bIdx < pe_blocksPerChunk[cu][brick_flag_idx]; bIdx++, index += block_size_in_bytes) {
                uint32_t block_size = block_size_in_bytes;
                if (index + block_size > pe_sizeOfChunk[cu][brick_flag_idx]) {
                    block_size = pe_sizeOfChunk[cu][brick_flag_idx] - index;
                }
                uint32_t no_sub_block = (block_size - 1) / 32768 + 1;
                for(uint32_t sub_idx = 0; sub_idx < no_sub_block; sub_idx++){

                    uint32_t compressed_size = (h_compressSize[cu][flag].data())[sub_idx + bIdx * compress_block_size/32];
                    if(cu == 0){
                        std::memcpy(&out1[outIdx_1], &h_buf_zlibout[cu][flag].data()[bIdx* block_size_in_bytes*33/32 + sub_idx*32768*33/32],
                                compressed_size);
                        outIdx_1 += compressed_size;
                        compress_size_out_1[compress_size_index_1++]= compressed_size;
                    }
                    else if(cu == 1){
                        std::memcpy(&out2[outIdx_2], &h_buf_zlibout[cu][flag].data()[bIdx* block_size_in_bytes*33/32 + sub_idx*32768*33/32],
                                compressed_size);
                        outIdx_2 += compressed_size;
                        compress_size_out_2[compress_size_index_2++]= compressed_size;
                        }
                    else if(cu == 2){
                        std::memcpy(&out3[outIdx_3], &h_buf_zlibout[cu][flag].data()[bIdx* block_size_in_bytes*33/32 + sub_idx*32768*33/32],
                                compressed_size);
                        outIdx_3 += compressed_size;
                        compress_size_out_3[compress_size_index_3++]= compressed_size;
                        }
                    else if(cu == 3){
                        std::memcpy(&out4[outIdx_4], &h_buf_zlibout[cu][flag].data()[bIdx* block_size_in_bytes*33/32 + sub_idx*32768*33/32],
                                compressed_size);
                        outIdx_4 += compressed_size;
                        compress_size_out_4[compress_size_index_4++]= compressed_size;
                        }
                }
            }
        }
    }
    //printf("finish \n");
    //delete(buffer_total_in[0]);
    //delete(buffer_total_in[1]);
    //delete(buffer_total_in[2]);
    //delete(buffer_total_in[3]);
    //buffer_total_in[0] = nullptr;
    //buffer_total_in[1] = nullptr;
    //buffer_total_in[2] = nullptr;
    //buffer_total_in[3] = nullptr;
    }
    // zlib special block based on Z_SYNC_FLUSH
    int xarg = 0;
    //out[outIdx + xarg++] = 0x01;
    //out[outIdx + xarg++] = 0x00;
    //out[outIdx + xarg++] = 0x00;
    //out[outIdx + xarg++] = 0xff;
    //out[outIdx + xarg++] = 0xff;
    //outIdx += xarg;
    //printf("total compressed size %d \n",outIdx);

    return (outIdx_1 + outIdx_2 + outIdx_3 + outIdx_4);
} // Overlap end
