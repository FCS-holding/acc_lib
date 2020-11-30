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
#include <fstream>
#include <vector>
#include "cmdlineparser.h"
//#include "license.h" 
#include <regex>
#include <iterator>
#include <string>
#include <stdlib.h>  
#include <libgen.h>
#include "fczlib_compress.hpp"

#define MAX_FILE_NUM 1024
#define BLOCK_MEM_SIZE 32768
//void fczlib_init(uint32_t file_size){
//    uint32_t compress_block_size;
//    if (file_size <= 1048576*3 ) 
//        compress_block_size = 32;
//    else if(file_size > 1024*1024*3 && file_size <= 1024*1024*16)
//        compress_block_size = 64;
//    else if(file_size > 1048576*16 && file_size <= 240*1024*1024)//3-240MB
//        compress_block_size = 256;
//    else
//        compress_block_size = 1024;
//    
//    std::string compress_bin = "kernel_top.xclbin";
//    
//    xil_zlib(compress_bin,compress_block_size);
//}
//void fczlib_compress(
//   uint8_t * in,
//   uint32_t compress_config,
//   uint8_t  compress_file_num,//max 1024
//   uint32_t * in_file_size,
//   uint8_t * out,
//   uint32_t * out_file_size
//){
//    //printf("file num %d \n", compress_file_num);
//    //printf("file size %d \n", in_file_size[0]);
//    std::chrono::duration<double, std::nano> compress_API_time_ns_1(0);
//    uint32_t * data_block_size = (uint32_t *)malloc(sizeof(uint32_t)*1024*128);
//    uint32_t * compress_size = (uint32_t *)malloc(sizeof(uint32_t)*1024*128);
//    uint32_t block_num_file[1024];
//    uint32_t data_block_index = 0;
//    uint32_t compress_block_size = 1024;
//    uint32_t total_file_size = 0;
//    for(uint16_t file_cnt = 0; file_cnt < compress_file_num; file_cnt++){
//        uint32_t tmp_file_size = in_file_size[file_cnt];
//        total_file_size += tmp_file_size;
//        block_num_file[file_cnt] = (tmp_file_size - 1)/32768 + 1;
//        for(uint32_t idx = 0; idx < tmp_file_size; idx += 32768){
//            if(idx + 32768 > tmp_file_size){
//                data_block_size[data_block_index] = tmp_file_size - idx;
//                }
//            else{
//                data_block_size[data_block_index] = 32768;
//                }
//            data_block_index++;
//            }
//        }
//
//    if (total_file_size <= 1048576*3 ) 
//        compress_block_size = 32;
//    else if(total_file_size > 1024*1024*3 && total_file_size <= 1024*1024*16)
//        compress_block_size = 64;
//    else if(total_file_size > 1048576*16 && total_file_size <= 240*1024*1024)//3-240MB
//        compress_block_size = 256;
//    else
//        compress_block_size = 1024;
//    auto compress_API_start = std::chrono::high_resolution_clock::now();
//    compress(in, out, compress_block_size, data_block_index, data_block_size, compress_size);
//    auto compress_API_end = std::chrono::high_resolution_clock::now();
//    uint32_t compress_block_index = 0;
//    for(uint32_t x = 0; x < compress_file_num; x++){
//        for(uint32_t t = 0; t < block_num_file[x]; t++){
//            out_file_size[x] += compress_size[compress_block_index];
//            compress_block_index++;
//        } 
//    }
//    auto duration = std::chrono::duration<double, std::nano>(compress_API_end - compress_API_start);
//    compress_API_time_ns_1 += duration;
//    float throughput_in_mbps_1 = (float)total_file_size * 1000 / compress_API_time_ns_1.count();
//    std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;
//    printf("\n");
//}
//void fczlib_release(){
//    xil_zlib_dec();
//    }
void xil_compress_top(std::string& compress_mod, std::string& single_bin, std::string& compress_out, uint32_t mod_sel) {
    

    std::ifstream inFile(compress_mod.c_str(), std::ifstream::binary);
    if (!inFile) {
        std::cout << "Unable to open file \'" << compress_mod << "\'" << endl;
        exit(1);
    }
    //=============================================
    //read file list and get the input size and compress_block_size
    std::string line;
    std::string file_in_name[MAX_FILE_NUM];
    std::string file_org_name[MAX_FILE_NUM];
    int file_num = 0;
    int tmp_file_size;
    uint32_t total_file_size = 0;
    uint32_t real_file_size = 0;
    uint32_t total_block_num = 0;
    uint32_t block_num_count = 0;
    uint32_t input_file_size[1024];
    uint32_t output_file_size[1024];
    for(int i = 0; i < 1024; i++){
        output_file_size[i] = 0;
        }
    int file_block_num[MAX_FILE_NUM];//up to 1024 files
    if(mod_sel == 1){
        while(std::getline(inFile, line)){
            std::ifstream inFile_1(line.c_str(), std::ifstream::binary);
            if(inFile){
                file_in_name[file_num] = line.c_str();
                char *path_buffer = const_cast<char *>(line.c_str());
                file_org_name[file_num] = basename(path_buffer);
                tmp_file_size = get_file_size(inFile_1);
                input_file_size[file_num] = tmp_file_size;
                real_file_size += tmp_file_size;
                int tmp_block_num = (tmp_file_size - 1) / BLOCK_MEM_SIZE + 1;
                block_num_count += tmp_block_num;
                total_file_size += tmp_block_num * BLOCK_MEM_SIZE;
                total_block_num += tmp_block_num;
                file_block_num[file_num] = tmp_block_num;
                file_num++;
                inFile_1.close();
                if(file_num >= MAX_FILE_NUM){
                    printf("too many files \n");
                    break;
                    }
            }
        }
    inFile.close();
    }

    //=============================================
    //uint32_t input_size = get_file_size(inFile);
    uint32_t input_size;
    if(mod_sel == 1){
        input_size = total_file_size;
    }
    else{
        input_size = get_file_size(inFile);
        }
    uint32_t compress_block_size;
    if (input_size <= 1048576*3 ) 
        compress_block_size = 32;
    else if(input_size > 1024*1024*3 && input_size <= 1024*1024*16)
        compress_block_size = 64;
    else if(input_size > 1048576*16 && input_size <= 240*1024*1024)//3-240MB
        compress_block_size = 256;
    else
        compress_block_size = 1024;

    //=======================================================================
    //read file to memory
    //=======================================================================
    uint32_t total_block_num_tmp = (input_size - 1)/ 32768 + 1;
    std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_format(total_block_num_tmp * 32768);
    uint32_t tmp_output_file_size = (input_size/8)*9 + 1;
    std::vector<uint8_t, aligned_allocator<uint8_t> > zlib_out(tmp_output_file_size);

    uint32_t pre_file_block_num = 0;
    if(mod_sel == 1){
        for(int i = 0; i < file_num; i++){
            std::ifstream inFile(file_in_name[i], std::ifstream::binary);
            inFile.read((char *)(zlib_format.data() + pre_file_block_num), input_file_size[i]);
            pre_file_block_num += file_block_num[i] * BLOCK_MEM_SIZE;
            inFile.close();
        }
    }
    else{
        inFile.read((char*)zlib_format.data(), input_size);
        input_file_size[file_num] = input_size;
        file_num++;
        inFile.close();
    }
    //=======================================================================
    //finish reading file
    //=======================================================================
    //compress_block_size = 32;
    std::string lz_compress_in = compress_mod;
    std::string lz_compress_out = compress_out;

    //init default bin kernel_top.xclbin 
    fczlib_init("kernel_top.xclbin",input_size);
    //fczlib_init();
    std::cout << std::fixed << std::setprecision(2) << "E2E Throughput(MB/s)\t:";
 
    // Call ZLIB compression
    //uint32_t enbytes = xlz->compress_file(lz_compress_in, lz_compress_out, input_size, compress_block_size, mod_sel);
    //uint32_t enbytes = compress_file(lz_compress_in, lz_compress_out, input_size, compress_block_size, mod_sel);
    //call fpga kernel executing
    fczlib_compress(
            zlib_format.data(),
            input_file_size,
            zlib_out.data(),
            output_file_size,
            file_num,
            1
            );
    uint32_t enbytes = 0;
    for(int j = 0; j < file_num; j++){
        enbytes += output_file_size[j];
        }
    printf("test step 1 \n");
    //=======================================================================
    //write data to file
    //=======================================================================
    uint32_t data_read_index = 0;
    for(int i = 0; i < file_num; i++){
        std::string out_file_name;
        if(mod_sel == 1){
            out_file_name = file_in_name[i]; 
            if(!compress_out.empty()){
                std::string tmp_file_name;
                tmp_file_name = compress_out + "/" + file_org_name[i] + ".zlib";
                out_file_name = tmp_file_name;
                }
            else{
                out_file_name = out_file_name + ".zlib";
            }
        }
        else{
            out_file_name = compress_out;
        }
        std::ofstream outFile(out_file_name, std::ofstream::binary);
        //write header
        outFile.put(120);
        outFile.put(1);
        //for(int j = 0; j < file_block_num[i]; j++){
            outFile.write((char *)(zlib_out.data() + data_read_index),output_file_size[i]);
            data_read_index += output_file_size[i];
        //} 
        
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
    printf("test step 2 \n");
    //=================================================
    //
    //=================================================
    std::cout.precision(3);
    std::cout << std::fixed << std::setprecision(2) << std::endl;
    // extract data from csv file
//    ifstream in("profile_summary.csv");
//    if(in) {
//        char str[1000];
//        double max_time = 0.0;
//        while(in) {
//            in.getline(str, 1000);
//            //zlib_top,1,0.64748,0.64748,0.64748,0.64748,
//            string s = str;
//            cout << " s= " << str << endl; 
//            std::regex self_regex("^zlib_top",
//                                  std::regex_constants::ECMAScript | std::regex_constants::icase);
//            if (std::regex_search(s, self_regex)) {
//                    std::cout << "match zlib_top" << std::endl;
//                std::regex sub_regex(",(.*),(.*),(.*),(.*),(.*),",
//                                     std::regex_constants::ECMAScript | std::regex_constants::icase);
//                std::smatch m;
//                auto pos = s.cbegin();
//                auto end = s.cend();
//                for (; std::regex_search(pos, end, m, sub_regex); pos = m.suffix().first) {
//                    std::cout << m.str(2) << std::endl;
//                    if(max_time < atof(m.str(2).c_str())) {
//                        max_time = atof(m.str(2).c_str());
//                    }
//                }  
//            }
//        }
//                    std::cout << "max time = " << max_time << std::endl;
//        std::cout << "Kernel Throughput(MB/s)\t:" << (double)input_size / max_time << std::endl;
//    }
//    in.close();
    fczlib_release();
    printf("test step 3 \n");
    //xil_zlib_dec();
    //release();

    std::cout << "Compression Rate\t:" << (double)input_size / enbytes << std::endl
              << std::fixed << std::setprecision(3) << "File Size(MB)\t\t:" << (double)input_size / 1000000 << std::endl
              << "File Name\t\t:" << lz_compress_in << std::endl;
    if(mod_sel == 0){
        std::cout << "Output Location\t\t:" << lz_compress_out.c_str() << std::endl;
    }
}
int main(int argc, char* argv[]) {
    if(argc < 2) {
        printf("Usage:\n");
        printf("<input>        : Input file or file list\n");
        printf("<output>       : (Optional) Output file, default name is \'<input>.zlib\'\n");
        printf("<kernel.xclbin>: (Optional) xclbin file, default name is \'kernel_top.xclbin\'\n");
        return 1;
    }
    //sda::utils::CmdLineParser parser;
    //parser.addSwitch("--xclbin", "-cx", "XCLBIN", "compress");
    //parser.parse(argc, argv);
    //std::string compress_bin = parser.value("xclbin");
    std::string mod_sel = argv[1];
    uint32_t mode = 0;
    std::string compress_mod;
    std::string compress_out;
    std::string compress_bin;
    if(mod_sel == "-l"){
        compress_mod = argv[2];
        //char *path_buffer = const_cast<char *>(compress_mod.c_str());
//        printf("File name with extension :%s\n",basename(path_buffer));
        //compress_out = string(basename(path_buffer)) + ".zlib";
        compress_bin = "kernel_top.xclbin";
        if(argc > 3) {
            compress_out = argv[3];
        }
        if(argc > 4) {
            compress_bin = argv[4];
        }
        mode = 1;
    }
    else{
        compress_mod = argv[1];
        char *path_buffer = const_cast<char *>(compress_mod.c_str());
//        printf("File name with extension :%s\n",basename(path_buffer));
        compress_out = string(basename(path_buffer)) + ".zlib";
        compress_bin = "kernel_top.xclbin";
        if(argc > 2) {
            compress_out = argv[2];
        }
        if(argc > 3) {
            compress_bin = argv[3];
        }
        mode = 0; 
    }
    //Feature feature = FALCON_FCZIP;
    //fc_license_init(); 
    //fc_license_checkout(feature, 1);
    printf("\n"); 
    if (!compress_mod.empty()) xil_compress_top(compress_mod, compress_bin, compress_out, mode);
    return 0;
    //fc_license_checkin(feature);
}
