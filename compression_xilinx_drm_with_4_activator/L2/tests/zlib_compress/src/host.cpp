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

#include <CL/cl2.hpp>
#include <CL/cl_ext_xilinx.h>
#include <xclhal2.h>
#include "accelize/drm.h"

#define MAX_FILE_NUM 1024
#define BLOCK_MEM_SIZE 32768
#define DRM_BASE_ADDRESS 0x1800000
using namespace Accelize::DRM;
xclDeviceHandle boardHandler;
/*
 * DRMLib Read Callback Function
 */
int32_t drm_read_callback(uint32_t addr, uint32_t *value)
{  
    int ret = (int)xclRead(boardHandler, XCL_ADDR_KERNEL_CTRL, DRM_BASE_ADDRESS+addr, value, 4);
    if(ret <= 0) {
        std::cout << __FUNCTION__ << ": Unable to read from the fpga ! ret = " << ret << std::endl;
        return 1;
    }
    return 0;
}

/*
 * DRMLib Write Callback Function
 */
int32_t drm_write_callback(uint32_t addr, uint32_t value)
{
    int ret = (int)xclWrite(boardHandler, XCL_ADDR_KERNEL_CTRL, DRM_BASE_ADDRESS+addr, &value, 4);
    if(ret <= 0) {
        std::cout << __FUNCTION__ << ": Unable to write to the fpga ! ret=" << ret << std::endl;
        return 1;
    }
    return 0;
}
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
    int file_num = 0;
    int tmp_file_size;
    uint32_t total_file_size = 0;
    uint32_t real_file_size = 0;
    uint32_t total_block_num = 0;
    uint32_t block_num_count = 0;
    if(mod_sel == 1){
        while(std::getline(inFile, line)){
            std::ifstream inFile_1(line.c_str(), std::ifstream::binary);
            if(inFile){
                file_in_name[file_num] = line.c_str();
                tmp_file_size = get_file_size(inFile_1);
                real_file_size += tmp_file_size;
                int tmp_block_num = (tmp_file_size - 1) / BLOCK_MEM_SIZE + 1;
                block_num_count += tmp_block_num;
                total_file_size += tmp_block_num * BLOCK_MEM_SIZE;
                total_block_num += tmp_block_num;
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
    if (input_size <= 1048576*4 ) 
        compress_block_size = 32;
    else if(input_size > 1024*1024*4 && input_size <= 1024*1024*16)
        compress_block_size = 64;
    else if(input_size > 1048576*16 && input_size <= 240*1024*1024)//3-240MB
        compress_block_size = 128;
    else
        compress_block_size = 1024;


    //compress_block_size = 32;
    std::string lz_compress_in = compress_mod;
    std::string lz_compress_out = compress_out;
//    lz_compress_out = lz_compress_out + ".zlib";

    // Xilinx ZLIB object
    xil_zlib* xlz;
    xlz = new xil_zlib(single_bin, 0, compress_block_size, input_size);
    //printf("after init !!! \n");
    //xlz = new xil_zlib(single_bin, 0, 1024, 51220480);

    // For compression m_bin_flow = 0
    xlz->m_bin_flow = 0;
    //=================================================
    if(xclProbe() < 1) {
        std::cout << "[ERROR] xclProbe failed ..." << std::endl;
        //return -1;
    }
    boardHandler = xclOpen(0, "xclhal2_logfile.log", XCL_ERROR);
    if(boardHandler == NULL) {
        std::cout << "[ERROR] xclOpen failed ..." << std::endl;
        //return -1;
    }

    //printf("block size in kB\t:%d\n",compress_block_size);

    // Call ZLIB compression
    //uint32_t enbytes = 0;
//ACCELIZE DRMLIB CODE AREA START      
    DrmManager *pDrmManag = new DrmManager(
        std::string("conf.json"),
        std::string("cred.json"),
        [&]( uint32_t  offset, uint32_t * value) {      /*Read DRM register*/
            return  drm_read_callback(offset, value);
        },
        [&]( uint32_t  offset, uint32_t value) {        /*Write DRM register*/
            return drm_write_callback(offset, value);
        },
        [&]( const  std::string & err_msg) {
           std::cerr  << err_msg << std::endl;
        }
    );
    std::cout << "[DRMLIB] Start Session .." << std::endl;
    pDrmManag->activate();
    //printf("start to compress !! \n");
    //ACCELIZE DRMLIB CODE AREA STOP
    std::cout << std::fixed << std::setprecision(2) << "E2E Throughput(MB/s)\t:";
    uint32_t enbytes = xlz->compress_file(lz_compress_in, lz_compress_out, input_size, compress_block_size, mod_sel);

    std::cout.precision(3);
    std::cout << std::fixed << std::setprecision(2) << std::endl;
    //ACCELIZE DRMLIB CODE AREA START
    std::cout << "[DRMLIB] Stop Session .." << std::endl;
    pDrmManag->deactivate();
    xclClose(boardHandler);
    //ACCELIZE DRMLIB CODE AREA STOP
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
    //fc_license_checkin(feature);
}
