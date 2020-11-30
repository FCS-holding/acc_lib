CFLAGS= -c -fPIC -std=c++0x -m64 -g -DBLAZE_TEST
ifeq ($(GPU),)
ifeq ($(XILINX_OPENCL_DIR),)
$(error XILINX_OPENCL_DIR not set)
endif
PP=$(XILINX_OPENCL_DIR)/lnx64/tools/gcc/bin/g++
CC=$(XILINX_OPENCL_DIR)/lnx64/tools/gcc/bin/gcc

COMPILE= $(CFLAGS) \
				 -I$(XILINX_OPENCL_DIR)/runtime/include/1_2

#LINK= -L$(XILINX_OPENCL_DIR)/runtime/lib/x86_64 -lOpenCL
LINK= -L$(XILINX_OPENCL_DIR)/runtime/lib/x86_64 -lxilinxopencl \
			-lpthread -lm

else
CUDA_DIR=/usr/local/cuda

PP=g++
CC=gcc

COMPILE= $(CFLAGS) -DGPU \
				 -I$(CUDA_DIR)/include

LINK= -L$(CUDA_DIR)/lib64 -lOpenCL \
			-lpthread -lm
endif

all: host

host: host.o gradient.o
	$(PP) -o $@ host.o gradient.o $(LINK)

%.o: %.cpp Makefile
	$(PP) $(COMPILE) $< -o $@

clean:
	rm -rf host
	rm -rf *.o
