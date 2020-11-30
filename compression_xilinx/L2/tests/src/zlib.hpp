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
#pragma once

#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <math.h>
#include <time.h>
#include <string>
#include <fstream>
#include <thread>
#include "xcl2.hpp"
#include "zlib_config.hpp"

#define PARALLEL_ENGINES 8
#define C_COMPUTE_UNIT 4 
#define D_COMPUTE_UNIT 1 
#define H_COMPUTE_UNIT 4 
#define T_COMPUTE_UNIT 4 
#define Z_COMPUTE_UNIT 4 
#define MAX_CCOMP_UNITS C_COMPUTE_UNIT
#define MAX_DDCOMP_UNITS D_COMPUTE_UNIT

// Default block size
//#define BLOCK_SIZE_IN_KB 1024
//#define BLOCK_SIZE_IN_KB 32

// Maximum host buffer used to operate
// per kernel invocation
//#define HOST_BUFFER_SIZE (PARALLEL_ENGINES * BLOCK_SIZE_IN_KB * 1024)

// Value below is used to associate with
// Overlapped buffers, ideally overlapped
// execution requires 2 resources per invocation
#define OVERLAP_BUF_COUNT 2

#define MAX_CHUNK 16
// Maximum number of blocks based on host buffer size
//#define MAX_NUMBER_BLOCKS (HOST_BUFFER_SIZE / (BLOCK_SIZE_IN_KB * 1024))
//#define MAX_NUMBER_BLOCKS (HOST_BUFFER_SIZE / (32 * 1024))

//int validate(std::string& inFile_name, std::string& outFile_name);

uint32_t get_file_size(std::ifstream& file);

//class xil_zlib {
//   public:
    int init(const std::string& binaryFile, uint8_t flow);
    int release();
    //void cu_thread_func(uint8_t* in, uint8_t* out, uint32_t input_size, uint32_t outIdx, uint32_t host_buffer_size, uint8_t queue_idx, uint32_t cu, int flag, uint32_t brick, uint32_t itr, uint32_t total_chunks,int completed_bricks, uint32_t overlap_buf_count,uint32_t block_size_in_bytes, uint32_t block_size_in_kb, uint32_t * sizeOfChunk, uint32_t * blocksPerChunk);
    //uint32_t compress(uint8_t* in, uint8_t* out, uint32_t actual_size, uint32_t  compress_block_size);
    uint32_t compress(uint8_t* in, uint8_t* out, uint32_t compress_block_size,
                            uint32_t data_block_num,
                            uint32_t * data_block_size_in,
                            uint32_t * compress_size_out);
    //uint32_t decompress(uint8_t* in, uint8_t* out, uint32_t actual_size, int cu_run);
    uint32_t compress_file(std::string& inFile_name, std::string& outFile_name, uint64_t input_size,uint32_t compress_block_size, uint32_t mod_sel);
    //uint32_t decompress_file(std::string& inFile_name, std::string& outFile_name, uint64_t input_size, int cu_run);
    //uint64_t get_event_duration_ns(const cl::Event& event);
    // Binary flow compress/decompress
    void xil_zlib(const std::string& binaryFile, uint32_t compress_block_size);
    void xil_zlib_dec();

    // Decompress Device Buffers
    //cl::Buffer* buffer_dec_input[MAX_DDCOMP_UNITS];
    //cl::Buffer* buffer_dec_zlib_output[MAX_DDCOMP_UNITS];
    //cl::Buffer* buffer_dec_compress_size[MAX_DDCOMP_UNITS];
//};
