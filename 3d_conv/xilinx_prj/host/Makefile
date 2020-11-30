
INCLUDE_PATH = -I$(XILINX_XRT)/include
INCLUDE_PATH += -I ../src -I ./ -I ../include
LIBRARY_PATH = -L/opt/xilinx/xrt_2.3.1/lib -L.
LIBRARY_PATH += -L/lib64 
LIBRARY_PATH += -L$(XILINX_SDX)/lib/lnx64.o 
LIBRARY_PATH += -L$(XILINX_VITIS)/lib/lnx64.o 
LIBRARY_PATH += -L$(XILINX_SDX)/runtime/lib/x86_64
LIBRARY = -lxilinxopencl -lstdc++ -std=c++11 -lstdc++ -lkernel 
CONFIGS = -Wall -O3 -D MCC_ACC -Wl,-rpath=./ -fPIC -D MCC_ACC_H_FILE=\"krnl_conv_3d.h\"
SRC = host.cpp
OUT = -o conv_3d_app_acc

host:
	xcpp $(INCLUDE_PATH) $(LIBRARY_PATH) $(LIBRARY) $(CONFIGS) $(SRC) $(OUT)

clean:
	rm -rf conv_3d_app_acc *.so
