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


//private:
cl::Program* m_program;
cl::Context* m_context;
cl::CommandQueue* m_q[C_COMPUTE_UNIT * OVERLAP_BUF_COUNT];
//cl::CommandQueue* m_q_dec[D_COMPUTE_UNIT];

bool m_bin_flow;
// Kernel declaration
cl::Kernel* zlib_kernel[Z_COMPUTE_UNIT];

// Compression related
std::vector<uint8_t, aligned_allocator<uint8_t> > h_buf_in_s[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
std::vector<uint8_t, aligned_allocator<uint8_t> > h_buf_zlibout[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
std::vector<uint32_t, aligned_allocator<uint32_t> > h_blksize[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
std::vector<uint32_t, aligned_allocator<uint32_t> > h_compressSize[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
// Buffers related to Dynamic Huffman
// Literal & length frequency tree
std::vector<uint32_t, aligned_allocator<uint32_t> > h_dyn_ltree_freq[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
// Distance frequency tree
std::vector<uint32_t, aligned_allocator<uint32_t> > h_dyn_dtree_freq[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
// Bit Length frequency
std::vector<uint32_t, aligned_allocator<uint32_t> > h_dyn_bltree_freq[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

// Literal Codes
std::vector<uint32_t, aligned_allocator<uint32_t> > h_dyn_ltree_codes[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
// Distance Codes
std::vector<uint32_t, aligned_allocator<uint32_t> > h_dyn_dtree_codes[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
// Bit Length Codes
std::vector<uint32_t, aligned_allocator<uint32_t> > h_dyn_bltree_codes[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

// Literal Bitlength
std::vector<uint32_t, aligned_allocator<uint32_t> > h_dyn_ltree_blen[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
// Distance Bitlength
std::vector<uint32_t, aligned_allocator<uint32_t> > h_dyn_dtree_blen[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
// Bit Length Bitlength
std::vector<uint32_t, aligned_allocator<uint32_t> > h_dyn_bltree_blen[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

std::vector<uint32_t, aligned_allocator<uint32_t> > h_buff_max_codes[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

cl_mem_ext_ptr_t ext_buffer_input_s[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
cl_mem_ext_ptr_t ext_buffer_input[MAX_CHUNK][MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
cl_mem_ext_ptr_t ext_buffer_zlib_output[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
cl_mem_ext_ptr_t ext_buffer_compress_size[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
cl_mem_ext_ptr_t ext_buffer_inblk_size[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

// Device buffers
cl::Buffer* buffer_input_s[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
cl::Buffer* buffer_zlib_output[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
cl::Buffer* buffer_compress_size[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
cl::Buffer* buffer_inblk_size[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

cl::Buffer* buffer_dyn_ltree_freq[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
cl::Buffer* buffer_dyn_dtree_freq[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
cl::Buffer* buffer_dyn_bltree_freq[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

cl::Buffer* buffer_dyn_ltree_codes[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
cl::Buffer* buffer_dyn_dtree_codes[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
cl::Buffer* buffer_dyn_bltree_codes[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

cl::Buffer* buffer_dyn_ltree_blen[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
cl::Buffer* buffer_dyn_dtree_blen[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];
cl::Buffer* buffer_dyn_bltree_blen[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

cl::Buffer* buffer_max_codes[MAX_CCOMP_UNITS][OVERLAP_BUF_COUNT];

// Kernel names
std::vector<std::string> zlib_kernel_names = {"zlib_top:{zlib_top_kernel0}","zlib_top:{zlib_top_kernel1}","zlib_top:{zlib_top_kernel2}","zlib_top:{zlib_top_kernel3}"};

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
    //memcpy(a,b,size);
    memcpy(a,b,size);
    //mu.unlock();
    }
//void memcpy_t0(uint8_t *a, uint8_t *b, uint32_t size){
//    //mu.lock();
//    cpu_set_t cpu_mask;
//    CPU_ZERO(&cpu_mask);
//    CPU_SET(2, &cpu_mask);
//    sched_setaffinity(0, sizeof(cpu_set_t), &cpu_mask);
//    //memcpy(a,b,size);
//    memcpy(a,b,size);
//    //mu.unlock();
//    }
//void memcpy_t1(uint8_t *a, uint8_t *b, uint32_t size){
//    //mu.lock();
//    cpu_set_t cpu_mask;
//    CPU_ZERO(&cpu_mask);
//    CPU_SET(3, &cpu_mask);
//    sched_setaffinity(0, sizeof(cpu_set_t), &cpu_mask);
//    //memcpy(a,b,size);
//    memcpy(a,b,size);
//    //mu.unlock();
//    }
//void memcpy_t2(uint8_t *a, uint8_t *b, uint32_t size){
//    //mu.lock();
//    cpu_set_t cpu_mask;
//    CPU_ZERO(&cpu_mask);
//    CPU_SET(4, &cpu_mask);
//    sched_setaffinity(0, sizeof(cpu_set_t), &cpu_mask);
//    //memcpy(a,b,size);
//    memcpy(a,b,size);
//    //mu.unlock();
//    }
//void memcpy_t3(uint8_t *a, uint8_t *b, uint32_t size){
//    //mu.lock();
//    cpu_set_t cpu_mask;
//    CPU_ZERO(&cpu_mask);
//    CPU_SET(5, &cpu_mask);
//    sched_setaffinity(0, sizeof(cpu_set_t), &cpu_mask);
//    //memcpy(a,b,size);
//    memcpy(a,b,size);
//    //mu.unlock();
//    }

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

//uint32_t xil_zlib::compress_file(std::string& inFile_name, std::string& outFile_name, uint64_t input_size, uint32_t compress_block_size, uint32_t mod_sel) {
uint32_t compress_file(std::string& inFile_name, std::string& outFile_name, uint64_t input_size, uint32_t compress_block_size, uint32_t mod_sel) {
    std::chrono::duration<double, std::nano> compress_API_time_ns_1(0);
    std::ifstream infilelist(inFile_name.c_str(),std::ifstream::binary);
    std::string line;
    std::string file_in_name[MAX_FILE_NUM];
    std::string file_org_name[MAX_FILE_NUM];
    int file_num = 0;
    int file_block_num[MAX_FILE_NUM];//up to 1024 files
    uint32_t file_size_array[MAX_FILE_NUM];//up to 1024 files
    int tmp_file_size;
    uint32_t total_file_size = 0;
    uint32_t real_file_size = 0;
    uint32_t total_block_num = 0;
    uint32_t block_num_count = 0;
    uint32_t * data_block_size = (uint32_t *)malloc(sizeof(uint32_t)*1024*48);
    uint32_t * compress_size = (uint32_t *)malloc(sizeof(uint32_t)*1024*48);
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

    uint32_t total_block_num_tmp = (total_file_size - 1)/ 32768 + 1;
    std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_format(total_block_num_tmp * 32768);
    uint32_t tmp_output_file_size = (total_file_size/8)*9 + 1;
    std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_out(tmp_output_file_size);
    uint32_t pre_file_block_num = 0;
    if(mod_sel == 1){
        for(int i = 0; i < file_num; i++){
            std::ifstream inFile(file_in_name[i], std::ifstream::binary);
            inFile.read((char *)(zlib_format.data() + pre_file_block_num), file_size_array[i]);
            pre_file_block_num += file_block_num[i] * BLOCK_MEM_SIZE;
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

    //uint32_t sizeOfChunk[512];
    //uint32_t idx = 0;
    //for(int i = 0; i < 512; i++){
    //    sizeOfChunk[i] = 0;
    //    }
    //uint32_t host_buffer_size = PARALLEL_ENGINES*compress_block_size_t*1024;
    //uint8_t *in_data = zlib_format.data();
    //uint32_t total_chunk = 0;
    //for (uint32_t i = 0; i < total_file_size; i += host_buffer_size, idx++) {
    //    uint32_t chunk_size = host_buffer_size;
    //    if (chunk_size + i > input_size) {
    //        chunk_size = input_size - i;
    //    }
    //    sizeOfChunk[idx] = chunk_size;
    //}
    //total_chunk = idx;
    //uint32_t no_chunk = 0;
    //uint32_t CU_NUM = 1;
    //if(total_chunk > 2)
    //    CU_NUM = 4;
    //no_chunk = (total_chunk - 1)/8 + 1;
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
    auto compress_API_start = std::chrono::high_resolution_clock::now();
    // zlib Compress
    uint32_t enbytes = compress(zlib_format.data(), zlib_out.data(), compress_block_size_t,
                                total_block_num,
                                data_block_size,
                                compress_size
                        );
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
        for(int j = 0; j < file_block_num[i]; j++){
            outFile.write((char *)(zlib_out.data() + data_read_index),compress_size[block_index]);
            data_read_index += compress_size[block_index];
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
        enbytes = data_read_index;
    // Pack zlib encoded stream .gz file
    //zip(inFile_name, outFile, zlib_out.data(), enbytes);

    // Close file
    //inFile.close();
    //outFile.close();
    return enbytes;
}


// Constructor
//xil_zlib::xil_zlib(const std::string& binaryFileName, uint8_t flow, uint32_t compress_block_size, uint64_t input_size) {void xil_zlib(const std::string& binaryFileName, uint8_t flow, uint32_t compress_block_size, uint64_t input_size) {
void xil_zlib(const std::string& binaryFileName, uint32_t compress_block_size) {
    init(binaryFileName,0);

    // printf("C_COMPUTE_UNIT \n");
    uint32_t block_size_in_kb = compress_block_size;
    //uint32_t block_size_in_bytes = block_size_in_kb * 1024;
    uint32_t overlap_buf_count = OVERLAP_BUF_COUNT;
    uint32_t host_buffer_size = PARALLEL_ENGINES * block_size_in_kb * 1024;
    uint32_t HOST_BUFFER_SIZE = host_buffer_size;
    uint32_t MAX_NUMBER_BLOCKS =  (HOST_BUFFER_SIZE / (32 * 1024));
    host_buffer_size = ((host_buffer_size - 1) / block_size_in_kb + 1) * block_size_in_kb;
    //uint32_t t;
    for (int i = 0; i < MAX_CCOMP_UNITS; i++) {
        for (int j = 0; j < OVERLAP_BUF_COUNT; j++) {
            // Index calculation
            h_buf_in_s[i][j].resize(PARALLEL_ENGINES * HOST_BUFFER_SIZE);
            //h_buf_out[i][j].resize(PARALLEL_ENGINES * HOST_BUFFER_SIZE * 4);
            h_buf_zlibout[i][j].resize(PARALLEL_ENGINES * HOST_BUFFER_SIZE * 33/32);
            h_blksize[i][j].resize(MAX_NUMBER_BLOCKS);
            h_compressSize[i][j].resize(MAX_NUMBER_BLOCKS);
            //h_compressSize[i][j].resize(256);
            //h_dyn_ltree_freq[i][j].resize(PARALLEL_ENGINES * LTREE_SIZE);
            //h_dyn_dtree_freq[i][j].resize(PARALLEL_ENGINES * DTREE_SIZE);
            //h_dyn_bltree_freq[i][j].resize(PARALLEL_ENGINES * BLTREE_SIZE);
            
            h_dyn_ltree_codes[i][j].resize(PARALLEL_ENGINES * LTREE_SIZE);
            h_dyn_dtree_codes[i][j].resize(PARALLEL_ENGINES * DTREE_SIZE);
            h_dyn_bltree_codes[i][j].resize(PARALLEL_ENGINES * BLTREE_SIZE);
            h_dyn_ltree_blen[i][j].resize(PARALLEL_ENGINES * LTREE_SIZE);
            h_dyn_dtree_blen[i][j].resize(PARALLEL_ENGINES * DTREE_SIZE);
            h_dyn_bltree_blen[i][j].resize(PARALLEL_ENGINES * BLTREE_SIZE);

            h_buff_max_codes[i][j].resize(PARALLEL_ENGINES * MAXCODE_SIZE);
        }
    }
    for (uint32_t cu = 0; cu < C_COMPUTE_UNIT; cu++) {
        for (uint32_t flag = 0; flag < overlap_buf_count; flag++) {
            ext_buffer_input_s[cu][flag].obj = h_buf_in_s[cu][flag].data();
            ext_buffer_input_s[cu][flag].param = 0 ;
            ext_buffer_inblk_size[cu][flag].obj = h_blksize[cu][flag].data(); // Setting Obj and Param to Zero
            ext_buffer_inblk_size[cu][flag].param = 0;
            ext_buffer_zlib_output[cu][flag].obj = h_buf_zlibout[cu][flag].data(); // Setting Obj and Param to Zero
            ext_buffer_zlib_output[cu][flag].param = 0;
            ext_buffer_compress_size[cu][flag].obj = h_compressSize[cu][flag].data(); // Setting Obj and Param to Zero
            ext_buffer_compress_size[cu][flag].param = 0;
            #if PLATFORM == 250
            if(cu == 0) {
                ext_buffer_input_s[cu][flag].flags = XCL_MEM_DDR_BANK0;
                ext_buffer_inblk_size[cu][flag].flags = XCL_MEM_DDR_BANK0;
                ext_buffer_zlib_output[cu][flag].flags = XCL_MEM_DDR_BANK0;
                ext_buffer_compress_size[cu][flag].flags = XCL_MEM_DDR_BANK0;
            } else if(cu == 1) {
                ext_buffer_input_s[cu][flag].flags = XCL_MEM_DDR_BANK1;
                ext_buffer_inblk_size[cu][flag].flags = XCL_MEM_DDR_BANK1;
                ext_buffer_zlib_output[cu][flag].flags = XCL_MEM_DDR_BANK1;
                ext_buffer_compress_size[cu][flag].flags = XCL_MEM_DDR_BANK1;
            } else if(cu == 2) {
                ext_buffer_input_s[cu][flag].flags = XCL_MEM_DDR_BANK2;
                ext_buffer_inblk_size[cu][flag].flags = XCL_MEM_DDR_BANK2;
                ext_buffer_zlib_output[cu][flag].flags = XCL_MEM_DDR_BANK2;
                ext_buffer_compress_size[cu][flag].flags = XCL_MEM_DDR_BANK2;
            } else if(cu == 3) {
                ext_buffer_input_s[cu][flag].flags = XCL_MEM_DDR_BANK3;
                ext_buffer_inblk_size[cu][flag].flags = XCL_MEM_DDR_BANK3;
                ext_buffer_zlib_output[cu][flag].flags = XCL_MEM_DDR_BANK3;
                ext_buffer_compress_size[cu][flag].flags = XCL_MEM_DDR_BANK3;
            }
            #else
            if(cu == 0) {
                ext_buffer_input_s[cu][flag].flags =       bank[4*cu + 0];
                ext_buffer_inblk_size[cu][flag].flags =    bank[4*cu + 1];
                ext_buffer_zlib_output[cu][flag].flags =   bank[4*cu + 2];
                ext_buffer_compress_size[cu][flag].flags = bank[4*cu + 3];
            } else if(cu == 1) {
                ext_buffer_input_s[cu][flag].flags =        bank[4*cu + 0];
                ext_buffer_inblk_size[cu][flag].flags =     bank[4*cu + 1];
                ext_buffer_zlib_output[cu][flag].flags =    bank[4*cu + 2];
                ext_buffer_compress_size[cu][flag].flags =  bank[4*cu + 3];
            } else if(cu == 2) {
                ext_buffer_input_s[cu][flag].flags =       bank[4*cu + 0];
                ext_buffer_inblk_size[cu][flag].flags =    bank[4*cu + 1];
                ext_buffer_zlib_output[cu][flag].flags =   bank[4*cu + 2];
                ext_buffer_compress_size[cu][flag].flags = bank[4*cu + 3];
            } else if(cu == 3) {
                ext_buffer_input_s[cu][flag].flags =       bank[4*cu + 0];
                ext_buffer_inblk_size[cu][flag].flags =    bank[4*cu + 1];
                ext_buffer_zlib_output[cu][flag].flags =   bank[4*cu + 2];
                ext_buffer_compress_size[cu][flag].flags = bank[4*cu + 3];
            }

            #endif
        }
    }
    // Device buffer allocation
    for (uint32_t cu = 0; cu < C_COMPUTE_UNIT; cu++) {
        for (uint32_t flag = 0; flag < overlap_buf_count; flag++) {

            buffer_input_s[cu][flag] = new cl::Buffer(*m_context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                    host_buffer_size, &ext_buffer_input_s[cu][flag]);

            buffer_inblk_size[cu][flag] = new cl::Buffer(*m_context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                         384 * sizeof(uint32_t), &ext_buffer_inblk_size[cu][flag]);

            buffer_zlib_output[cu][flag] = new cl::Buffer(*m_context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                                          host_buffer_size *33/32, &ext_buffer_zlib_output[cu][flag]);

            buffer_compress_size[cu][flag] =
                new cl::Buffer(*m_context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, 384 * sizeof(uint32_t),
                               &ext_buffer_compress_size[cu][flag]);

        }
    }
}

// Destructor
void xil_zlib_dec() {
    release();
    uint32_t overlap_buf_count = OVERLAP_BUF_COUNT;
    //for (uint32_t t = 0; t < MAX_CHUNK; t++) {
    //    for (uint32_t cu = 0; cu < C_COMPUTE_UNIT; cu++) {
    //        for (uint32_t flag = 0; flag < overlap_buf_count; flag++) {
    //            //delete (buffer_input[t][cu][flag]);
    //        }
    //    }
    //}
    for (uint32_t cu = 0; cu < C_COMPUTE_UNIT; cu++) {
        for (uint32_t flag = 0; flag < overlap_buf_count; flag++) {
            delete (buffer_input_s[cu][flag]);
            //delete (buffer_lz77_output[cu][flag]);
            delete (buffer_zlib_output[cu][flag]);
            delete (buffer_compress_size[cu][flag]);
            delete (buffer_inblk_size[cu][flag]);

            //delete (buffer_dyn_ltree_freq[cu][flag]);
            //delete (buffer_dyn_dtree_freq[cu][flag]);
            //delete (buffer_dyn_bltree_freq[cu][flag]);

            //delete (buffer_dyn_ltree_codes[cu][flag]);
            //delete (buffer_dyn_dtree_codes[cu][flag]);
            //delete (buffer_dyn_bltree_codes[cu][flag]);

            //delete (buffer_dyn_ltree_blen[cu][flag]);
            //delete (buffer_dyn_dtree_blen[cu][flag]);
            //delete (buffer_dyn_bltree_blen[cu][flag]);
        }
    }
}

//int xil_zlib::init(const std::string& binaryFileName, uint8_t flow) {
int init(const std::string& binaryFileName, uint8_t flow) {
    // The get_xil_devices will return vector of Xilinx Devices
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Creating Context and Command Queue for selected Device
    m_context = new cl::Context(device);

    for (uint8_t i = 0; i < C_COMPUTE_UNIT * OVERLAP_BUF_COUNT; i++) {
        m_q[i] = new cl::CommandQueue(*m_context, device, CL_QUEUE_PROFILING_ENABLE);
    }

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Found Device=" << device_name.c_str() << std::endl;

    auto fileBuf = xcl::read_binary_file(binaryFileName);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    devices.resize(1);
    m_program = new cl::Program(*m_context, devices, bins);

    if (flow == 0 || flow == 2) {
        for (int i = 0; i < T_COMPUTE_UNIT; i++) {
            zlib_kernel[i] = new cl::Kernel(*m_program,zlib_kernel_names[i].c_str());
        }
    }

    return 0;
}

//int xil_zlib::release() {
int release() {
    for (int i = 0; i < Z_COMPUTE_UNIT; i++) delete (zlib_kernel[i]);
    delete (m_program);

    for (uint8_t i = 0; i < C_COMPUTE_UNIT * OVERLAP_BUF_COUNT; i++) {
        delete (m_q[i]);
    }
    delete (m_context);

    return 0;
}

//uint32_t xil_zlib::compress(uint8_t* in, uint8_t* out, uint32_t compress_block_size,
uint32_t compress(uint8_t* in, uint8_t* out, uint32_t compress_block_size,
                            uint32_t data_block_num,
                            uint32_t * data_block_size_in,
                            uint32_t * compress_size_out
                            ) {
    //////printme("In compress \n");
    uint32_t block_size_in_kb = compress_block_size;
    uint32_t block_size_in_bytes = block_size_in_kb * 1024;
    uint32_t overlap_buf_count = OVERLAP_BUF_COUNT;
    uint32_t host_buffer_size = PARALLEL_ENGINES * compress_block_size * 1024;
    uint32_t input_size = data_block_num * 32768;

    // For example: Input file size is 12MB and Host buffer size is 2MB
    // Then we have 12/2 = 6 chunks exists
    // Calculate the count of total chunks based on input size
    // This count is used to overlap the execution between chunks and file
    // operations

    uint32_t total_chunks = (input_size - 1) / host_buffer_size + 1;
    //printf("total chunks %d \n", total_chunks);
    if (total_chunks < 2) overlap_buf_count = 1;

    // Find out the size of each chunk spanning entire file
    // For eaxmple: As mentioned in previous example there are 6 chunks
    // Code below finds out the size of chunk, in general all the chunks holds
    // HOST_BUFFER_SIZE except for the last chunk
    uint32_t sizeOfChunk[total_chunks];
    uint32_t blocksPerChunk[total_chunks];
    uint32_t idx = 0;
    uint32_t block_size_idx = 0;
    for (uint32_t i = 0; i < input_size; i += host_buffer_size, idx++) {
        uint32_t chunk_size = host_buffer_size;
        if (chunk_size + i > input_size) {
            chunk_size = input_size - i;
        }
        // Update size of each chunk buffer
        sizeOfChunk[idx] = chunk_size;
        // Calculate sub blocks of size BLOCK_SIZE_IN_KB for each chunk
        // 2MB(example)
        // Figure out blocks per chunk
        uint32_t nblocks = (chunk_size - 1) / block_size_in_bytes + 1;
        blocksPerChunk[idx] = nblocks;
    }

    // Counter which helps in tracking
    // Output buffer index
    uint32_t outIdx = 0;

    // Track the lags of respective chunks for left over handling
    int chunk_flags[total_chunks];
    int cu_order[total_chunks];

    // Finished bricks
    int completed_bricks = 0;

    int flag = 0;
    uint32_t lcl_cu = 0;

    uint8_t cunits = (uint8_t)C_COMPUTE_UNIT;
    uint8_t queue_idx = 0;
    uint32_t compress_size_index = 0;
    std::thread thread_memcpy;
    std::thread thread_memcpy0;
    std::thread thread_memcpy1;
    std::thread thread_memcpy2;
    std::thread thread_memcpy3;
    std::thread thread_dataout;
    //std::thread memin_thread[C_COMPUTE_UNIT];
    //auto kernel_start_0 = std::chrono::high_resolution_clock::now();
overlap:
    for (uint32_t brick = 0, itr = 0; brick < total_chunks; /*brick += C_COMPUTE_UNIT,*/ itr++, flag = !flag) {
        if (cunits > 1)
            queue_idx = flag * OVERLAP_BUF_COUNT * 2;
        else
            queue_idx = flag;

        if (total_chunks > 2)
            lcl_cu = C_COMPUTE_UNIT;
        else
            lcl_cu = 1;

        if (brick + lcl_cu > total_chunks) lcl_cu = total_chunks - brick;
        //auto kernel_start_0_1 = std::chrono::high_resolution_clock::now();
        //if(input_size > 12*1024*1024) {
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
                    //int policy;
                    //pthread_getschedparam(thread_memcpy.native_handle(), &policy, &sch);
                    //printf("thread priority %d \n",sch.sched_priority);
                    //sch.sched_priority = 99; 
                    //pthread_setschedparam(thread_memcpy.native_handle(), SCHED_RR, &sch);
                    //thread_memcpy.detach();
                    //
                    //test_read out
                    //
                    uint32_t index = 0;
                    uint32_t brick_flag_idx = brick - (C_COMPUTE_UNIT * overlap_buf_count - cu);
                    //==========================
                    //bind specified cpu
                    //==========================
                    //cpu_set_t cpuset;
                    //CPU_ZERO(&cpuset);
                    //CPU_SET(0,&cpuset);
                    //int rc = pthread_setaffinity_np(thread_memcpy.native_handle(),sizeof(cpu_set_t), &cpuset);
                    //CPU_SET(33,&cpuset);
                    //if(rc != 0){
                    //    printf("Error !!! \n");
                    //    }
                    //pthread_setaffinity_np(thread_dataout.native_handle(),sizeof(cpu_set_t), &cpuset);
                    // Copy the data from various blocks in concatinated manner
                    //uint32_t quartor_size = sizeOfChunk[brick + cu]/4;
                    if(compress_block_size >= 1024){
                    thread_memcpy = std::thread(memcpy_t, h_buf_in_s[cu][flag].data(), &in[(brick + cu) * host_buffer_size], sizeOfChunk[brick + cu]);
                    //thread_memcpy0 = std::thread(memcpy_t0, &h_buf_in_s[cu][flag].data()[0], &in[(brick + cu) * host_buffer_size], quartor_size);
                    //thread_memcpy1 = std::thread(memcpy_t1, &h_buf_in_s[cu][flag].data()[quartor_size], &in[(brick + cu) * host_buffer_size + quartor_size], quartor_size);
                    //thread_memcpy2 = std::thread(memcpy_t2, &h_buf_in_s[cu][flag].data()[quartor_size*2], &in[(brick + cu) * host_buffer_size + quartor_size*2], quartor_size);
                    //thread_memcpy3 = std::thread(memcpy_t3, &h_buf_in_s[cu][flag].data()[quartor_size*3], &in[(brick + cu) * host_buffer_size + quartor_size*3], quartor_size);
                    thread_dataout = std::thread(data_read_out,&outIdx,cu,brick_flag_idx,compress_block_size, block_size_in_bytes,out,h_buf_zlibout[cu][flag].data(),h_compressSize[cu][flag].data(),compress_size_out,&blocksPerChunk[0],&sizeOfChunk[0],&compress_size_index);
                    //data_read_out(&outIdx,cu,brick_flag_idx,compress_block_size,block_size_in_bytes,out,h_buf_zlibout[cu][flag].data(),h_compressSize[cu][flag].data(),compress_size_out,blocksPerChunk,sizeOfChunk,&compress_size_index);
                    //thread_dataout.detach();
                    thread_memcpy.join();
                    //thread_memcpy0.join();
                    //thread_memcpy1.join();
                    //thread_memcpy2.join();
                    //thread_memcpy3.join();
                    thread_dataout.join();
                    }
                    else{
                    for (uint32_t bIdx = 0; bIdx < blocksPerChunk[brick_flag_idx]; bIdx++, index += block_size_in_bytes) {
                        uint32_t block_size = block_size_in_bytes;
                        if (index + block_size > sizeOfChunk[brick_flag_idx]) {
                            block_size = sizeOfChunk[brick_flag_idx] - index;
                        }
                        uint32_t no_sub_block = (block_size - 1) / 32768 + 1;
                        //uint32_t no_sub_block =  1;
                        for(uint32_t sub_idx = 0; sub_idx < no_sub_block; sub_idx++){
                            uint32_t compressed_size = (h_compressSize[cu][flag].data())[sub_idx + bIdx * compress_block_size/32];
                            //uint32_t compressed_size = (h_compressSize[cu][flag].data())[bIdx];
                            //printf("cu %d \n",cu);
                            //printf("compress size index %d \n",sub_idx + bIdx * compress_block_size/32);
                            //printf("compressed_size %d \n",compressed_size);
                            //std::memcpy(&out[outIdx], &h_buf_zlibout[cu][flag].data()[bIdx* block_size_in_bytes*2 + sub_idx*32768*2],
                            //        compressed_size);
                            std::memcpy(&out[outIdx], &h_buf_zlibout[cu][flag].data()[bIdx* block_size_in_bytes*33/32 + sub_idx*32768*33/32],
                                    compressed_size);
                            //std::memcpy(&out[outIdx], &h_buf_zlibout[cu][flag].data()[bIdx* block_size_in_bytes*2 + sub_idx*32768*2],65536);
                            outIdx += compressed_size;
                            compress_size_out[compress_size_index++]= compressed_size;
                            //outIdx += 65536;
                        }
                    }
                    }
                } // If condition which reads huffman output for 0 or 1 location

                // Figure out block sizes per brick
                uint32_t idxblk = 0;
                for (uint32_t i = 0; i < sizeOfChunk[brick + cu]; i += 32768) {
                    uint32_t block_size = 32768;
                    block_size = data_block_size_in[block_size_idx++];    
                    //if(i + block_size > sizeOfChunk[brick + cu]){
                    //    block_size = sizeOfChunk[brick + cu] - i; 
                    //    }
                    (h_blksize[cu][flag]).data()[idxblk++] = block_size;
                    //printf("block in size %d %d \n",idxblk, block_size);
                
                }
                if(compress_block_size >= 1024){
                    if(itr < 2){
                        std::memcpy(h_buf_in_s[cu][flag].data(), &in[(brick + cu) * host_buffer_size], sizeOfChunk[brick + cu]);
                    }
                }
                else{
                        std::memcpy(h_buf_in_s[cu][flag].data(), &in[(brick + cu) * host_buffer_size], sizeOfChunk[brick + cu]);
                }
                // Set kernel arguments
                int narg = 0;
                //mu.lock();
                (zlib_kernel[cu])->setArg(narg++, *(buffer_input_s[cu][flag]));
                (zlib_kernel[cu])->setArg(narg++, *(buffer_inblk_size[cu][flag]));
                (zlib_kernel[cu])->setArg(narg++, block_size_in_kb);
                (zlib_kernel[cu])->setArg(narg++, sizeOfChunk[brick + cu]);
                (zlib_kernel[cu])->setArg(narg++, blocksPerChunk[brick]);
                (zlib_kernel[cu])->setArg(narg++, *(buffer_zlib_output[cu][flag]));
                (zlib_kernel[cu])->setArg(narg++, *(buffer_compress_size[cu][flag]));
                m_q[queue_idx + cu]->enqueueMigrateMemObjects({*(buffer_input_s[cu][flag]), *(buffer_inblk_size[cu][flag])},
                                                              0 /* 0 means from host*/);
                //// kernel write events update
                //// LZ77 Compress Fire Kernel invocation
                m_q[queue_idx + cu]->enqueueTask(*zlib_kernel[cu]);
                m_q[queue_idx + cu]->enqueueMigrateMemObjects(
                    {*(buffer_zlib_output[cu][flag]), *(buffer_compress_size[cu][flag])}, CL_MIGRATE_MEM_OBJECT_HOST);
                //mu.unlock();
            } // Internal loop runs on compute units
        //}
        //else{
        //    for (uint32_t cu = 0; cu < lcl_cu; cu++) {
        //        chunk_flags[brick + cu] = flag;
        //        cu_order[brick + cu] = cu;

        //        // Wait for read events
        //        if (itr >= 2) {
        //            // Wait on current flag previous operation to finish
        //            
        //            //printf("queue_idx: %d cu :%d\n", queue_idx, cu);
        //            m_q[queue_idx + cu]->finish();

        //            // Completed bricks counter
        //            completed_bricks++;

        //            uint32_t index = 0;
        //            uint32_t brick_flag_idx = brick - (C_COMPUTE_UNIT * overlap_buf_count - cu);

        //            // Copy the data from various blocks in concatinated manner
        //            for (uint32_t bIdx = 0; bIdx < blocksPerChunk[brick_flag_idx]; bIdx++, index += block_size_in_bytes) {
        //                uint32_t block_size = block_size_in_bytes;
        //                if (index + block_size > sizeOfChunk[brick_flag_idx]) {
        //                    block_size = sizeOfChunk[brick_flag_idx] - index;
        //                }
        //                uint32_t no_sub_block = (block_size - 1) / 32768 + 1;
        //                //uint32_t no_sub_block =  1;
        //                for(uint32_t sub_idx = 0; sub_idx < no_sub_block; sub_idx++){
        //                    uint32_t compressed_size = (h_compressSize[cu][flag].data())[sub_idx + bIdx * compress_block_size/32];
        //                    std::memcpy(&out[outIdx], &h_buf_zlibout[cu][flag].data()[bIdx* block_size_in_bytes*33/32 + sub_idx*32768*33/32],
        //                            compressed_size);
        //                    //std::memcpy(&out[outIdx], &h_buf_zlibout[cu][flag].data()[bIdx* block_size_in_bytes*2 + sub_idx*32768*2],65536);
        //                    //printf("cu %d \n",cu);
        //                    //printf("compress size index %d \n",sub_idx + bIdx * compress_block_size/32);
        //                    //printf("compressed_size %d \n",compressed_size);
        //                    compress_size_out[compress_size_index++]= compressed_size;
        //                    outIdx += compressed_size;
        //                    //outIdx += 65536;
        //                }
        //            }
        //        } // If condition which reads huffman output for 0 or 1 location

        //        // Figure out block sizes per brick
        //        uint32_t idxblk = 0;
        //        for (uint32_t i = 0; i < sizeOfChunk[brick + cu]; i += 32768) {
        //            uint32_t block_size = 32768;
        //            block_size = data_block_size_in[block_size_idx++];    
        //            //if(i + block_size > sizeOfChunk[brick + cu]){
        //            //    block_size = sizeOfChunk[brick + cu] - i; 
        //            //    }
        //            (h_blksize[cu][flag]).data()[idxblk++] = block_size; 
        //            //printf("block in size %d %d \n",idxblk, block_size);
        //        
        //        }
        //    }
        //    for (uint32_t cu = 0; cu < lcl_cu; cu++) {
        //        //printf("size of chunk %d \n", sizeOfChunk[brick + cu]); 
        //        // Set kernel arguments
        //        int narg = 0;
        //        (zlib_kernel[cu])->setArg(narg++, *(buffer_input[brick/8][cu][flag]));
        //        (zlib_kernel[cu])->setArg(narg++, *(buffer_inblk_size[cu][flag]));
        //        (zlib_kernel[cu])->setArg(narg++, block_size_in_kb);
        //        (zlib_kernel[cu])->setArg(narg++, sizeOfChunk[brick + cu]);
        //        (zlib_kernel[cu])->setArg(narg++, blocksPerChunk[brick]);
        //        (zlib_kernel[cu])->setArg(narg++, *(buffer_zlib_output[cu][flag]));
        //        (zlib_kernel[cu])->setArg(narg++, *(buffer_compress_size[cu][flag]));

        //    }
        //    for (uint32_t cu = 0; cu < lcl_cu; cu++) {
        //            m_q[queue_idx + cu]->enqueueMigrateMemObjects({*(buffer_input[brick/8][cu][flag]), *(buffer_inblk_size[cu][flag])},
        //                                                      0 /* 0 means from host*/);
        //    }
        //    for (uint32_t cu = 0; cu < lcl_cu; cu++) {
        //        m_q[queue_idx + cu]->enqueueTask(*zlib_kernel[cu]);
        //    }
        //    for (uint32_t cu = 0; cu < lcl_cu; cu++) {
        //        m_q[queue_idx + cu]->enqueueMigrateMemObjects(
        //            {*(buffer_zlib_output[cu][flag]), *(buffer_compress_size[cu][flag])}, CL_MIGRATE_MEM_OBJECT_HOST);
        //    }
        //} // Internal loop runs on compute units
            
        //}
        //auto kernel_start_0_6 = std::chrono::high_resolution_clock::now();
        //printf("test start 0.6 %u \n",kernel_start_0_6);

        if (total_chunks > 2)
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

    uint32_t leftover = total_chunks - completed_bricks;
    uint32_t stride = 0;

    if ((total_chunks < overlap_buf_count * C_COMPUTE_UNIT))
        stride = overlap_buf_count * C_COMPUTE_UNIT;
    else
        stride = total_chunks;

    // Handle leftover bricks
    for (uint32_t ovr_itr = 0, brick = stride - overlap_buf_count * C_COMPUTE_UNIT; ovr_itr < leftover;
         ovr_itr += C_COMPUTE_UNIT, brick += C_COMPUTE_UNIT) {
        lcl_cu = C_COMPUTE_UNIT;
        if (ovr_itr + lcl_cu > leftover) lcl_cu = leftover - ovr_itr;

        // Handle multiple bricks with multiple CUs
        for (uint32_t j = 0; j < lcl_cu; j++) {
            int cu = cu_order[brick + j];
            int flag = chunk_flags[brick + j];

            // Run over each block within brick
            uint32_t index = 0;
            uint32_t brick_flag_idx = brick + j;

            //////printme("blocksPerChunk %d \n", blocksPerChunk[brick]);
            // Copy the data from various blocks in concatinated manner
            for (uint32_t bIdx = 0; bIdx < blocksPerChunk[brick_flag_idx]; bIdx++, index += block_size_in_bytes) {
                uint32_t block_size = block_size_in_bytes;
                if (index + block_size > sizeOfChunk[brick_flag_idx]) {
                    block_size = sizeOfChunk[brick_flag_idx] - index;
                }
                uint32_t no_sub_block = (block_size - 1) / 32768 + 1;
                //uint32_t no_sub_block =  1;
                for(uint32_t sub_idx = 0; sub_idx < no_sub_block; sub_idx++){

                    uint32_t compressed_size = (h_compressSize[cu][flag].data())[sub_idx + bIdx * compress_block_size/32];
                    std::memcpy(&out[outIdx], &h_buf_zlibout[cu][flag].data()[bIdx * block_size_in_bytes*33/32 + 32768*sub_idx*33/32], compressed_size);
                    compress_size_out[compress_size_index++]= compressed_size;
                    outIdx += compressed_size;
                    //outIdx += 65536;
                }
            }
        }
    }

    // zlib special block based on Z_SYNC_FLUSH
    int xarg = 0;
    outIdx += xarg;

    return outIdx;
} // Overlap end
