g++ -O -shared -fPIC -DHLS_NO_XIL_FPO_LIB -o ./bin/libkernel.so /home/yuhang/test_acc_lib_compression/compression_xilinx_u50_4pe_32engine_opt_pcie_2/L2/tests/zlib_compress//src/fczlib_compress.cpp /home/yuhang/test_acc_lib_compression/compression_xilinx_u50_4pe_32engine_opt_pcie_2/L2/tests/src//zlib.cpp /home/yuhang/test_acc_lib_compression/compression_xilinx_u50_4pe_32engine_opt_pcie_2/common/libs/xcl2/xcl2.cpp /home/yuhang/test_acc_lib_compression/compression_xilinx_u50_4pe_32engine_opt_pcie_2/common/libs/cmdparser/cmdlineparser.cpp /home/yuhang/test_acc_lib_compression/compression_xilinx_u50_4pe_32engine_opt_pcie_2/common/libs/logger/logger.cpp /home/yuhang/test_acc_lib_compression/compression_xilinx_u50_4pe_32engine_opt_pcie_2/common/thirdParty/xxhash/xxhash.c -DPARALLEL_BLOCK=8 -I/home/yuhang/test_acc_lib_compression/compression_xilinx_u50_4pe_32engine_opt_pcie_2/L2/tests/zlib_compress/src/ -I/home/yuhang/test_acc_lib_compression/compression_xilinx_u50_4pe_32engine_opt_pcie_2/L2/include/ -I/home/yuhang/test_acc_lib_compression/compression_xilinx_u50_4pe_32engine_opt_pcie_2/L2/tests/src// -I/opt/xilinx/xrt/include/ -I/home/yuhang/test_acc_lib_compression/compression_xilinx_u50_4pe_32engine_opt_pcie_2/common/libs/xcl2/ -I/home/yuhang/test_acc_lib_compression/compression_xilinx_u50_4pe_32engine_opt_pcie_2/common/libs/cmdparser/ -I/home/yuhang/test_acc_lib_compression/compression_xilinx_u50_4pe_32engine_opt_pcie_2/common/libs/logger/ -I/home/yuhang/test_acc_lib_compression/compression_xilinx_u50_4pe_32engine_opt_pcie_2/common/thirdParty/xxhash/ -fmessage-length=0 -std=c++11 -Wno-unknown-pragmas -Wno-unused-label -pthread -L/opt/xilinx/xrt/lib/ -lxilinxopencl -pthread -lrt -Wno-unused-label -Wno-narrowing -std=c++11 -DVERBOSE -lstdc++  /home/yuhang/test_acc_lib_compression/compression_xilinx_u50_4pe_32engine_opt_pcie_2/license/license_pic.o /home/yuhang/test_acc_lib_compression/compression_xilinx_u50_4pe_32engine_opt_pcie_2/license/lm_new_pic.o -I /home/yuhang/test_acc_lib_compression/compression_xilinx_u50_4pe_32engine_opt_pcie_2/license/ -L /home/yuhang/test_acc_lib_compression/compression_xilinx_u50_4pe_32engine_opt_pcie_2/license/ -Wl,-rpath=/home/yuhang/test_acc_lib_compression/compression_xilinx_u50_4pe_32engine_opt_pcie_2/license/ -llmgr_trl_pic -lcrvs_pic -lsb_pic -lnoact_pic -llmgr_dongle_stub_pic -lpthread -ldl -fPIC
