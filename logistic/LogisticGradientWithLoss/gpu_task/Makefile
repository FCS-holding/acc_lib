ifeq ($(BLAZE_HOME),)
$(error BLAZE_HOME not set)
endif
ifeq ("$(wildcard $(BLAZE_HOME)/Makefile.config)","")
$(error BLAZE_HOME not set properly in environment)
endif

MANAGER_DIR=$(BLAZE_HOME)/manager
PLATFORM_DIR=$(BLAZE_HOME)/platforms/gpu-ocl
include $(BLAZE_HOME)/Makefile.config

#PP=g++
#CC=gcc
PP=$(XILINX_OPENCL_DIR)/lnx64/tools/gcc/bin/g++
CC=$(XILINX_OPENCL_DIR)/lnx64/tools/gcc/bin/gcc

CFLAGS= -shared -fPIC -std=c++0x -m64 -g 
COMPILE= $(CFLAGS) \
				 -I$(MANAGER_DIR)/src \
				 -I$(PLATFORM_DIR) \
				 -I$(BOOST_DIR)/include \
				 -I$(PROTOBUF_DIR)/include \
				 -I$(XILINX_OPENCL_DIR)/runtime/include/1_2
#-I$(CUDA_DIR)/include

LINK= -L$(XILINX_OPENCL_DIR)/runtime/lib/x86_64 -lOpenCL \
			-lpthread -lm
#LINK= -L$(CUDA_DIR)/lib64 -lOpenCL \
#			-lpthread -lm

ACC = Logistic

OBJS= $(addsuffix .so, $(ACC))

all: $(OBJS)

%.so: %.cpp Makefile
	$(PP) $(COMPILE) $(LINK) $< -o $@

clean:
	rm -rf $(OBJS)
