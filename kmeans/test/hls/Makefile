ifeq ($(BLAZE_HOME),)
$(error BLAZE_HOME not set)
endif
ifeq ("$(wildcard $(BLAZE_HOME)/Makefile.config)","")
$(error BLAZE_HOME not set properly in environment)
endif

MANAGER_DIR=$(BLAZE_HOME)/manager
PLATFORM_DIR=$(BLAZE_HOME)/platforms/xil_ocl
include $(BLAZE_HOME)/Makefile.config

PP=$(XILINX_OPENCL_DIR)/lnx64/tools/gcc/bin/g++
CC=$(XILINX_OPENCL_DIR)/lnx64/tools/gcc/bin/gcc

CFLAGS= -c -fPIC -std=c++0x -m64 -g -DHLS_SIM #-DSUPPORT_WIDE_BUS
COMPILE= $(CFLAGS) \
				 -I$(XILINX_OPENCL_DIR)/Vivado_HLS/include

LINK= -lpthread -lm

all: host

host: host.o kmeans.o
	$(PP) $(LINK) -o $@ host.o kmeans.o

%.o: %.cpp Makefile *.h
	$(PP) $(COMPILE) $< -o $@

clean:
	rm -rf host *.o
