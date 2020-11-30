SHELL = /bin/bash

ifndef XILINX_SDX
$(error Environment variable XILINX_SDX is required and should point to SDAccel install area)
endif

HOST_SRCS := main.cpp
OBJECTS := main.o

HOST = ./host_top
HOST_CFLAGS= -g -fPIC -std=c++0x -g -m64 \
             -DTARGET_DEVICE=\"${XDEVICE}\" -DFPGA_DEVICE -DC_KERNEL \
						 -I../../common/include \
		-I$(XILINX_SDX)/runtime/include/1_2 \
		-I$(XILINX_SDX)/Vivado_HLS/include

ifneq ($(DO_CSIM),)
	HOST_CFLAGS := $(HOST_CFLAGS) -DHLS_
endif

LINK= -L$(XILINX_SDX)/runtime/lib/x86_64 -lxilinxopencl\
		-lpthread -lm -ldl -lrt -lstdc++

KERNEL_SRCS = ../../kernel/kmeans.cpp
KERNEL_NAME = kmeans
KERNEL_DEFS = 
KERNEL_INCS = 
#KERNEL_FREQ = 200
#CLCC_OPT_LEVEL=-O3
#set target device for XCLBIN
#XDEVICE=F1
ifeq (${XDEVICE},)
    PLATFORM=xilinx:xil-accel-rd-ku115:4ddr-xpr:3.3
else ifeq (${XDEVICE},F1)
    PLATFORM=$(SDK_DIR)/SDAccel/aws_platform/xilinx_aws-vu9p-f1_4ddr-xpr-2pr_4_0/xilinx_aws-vu9p-f1_4ddr-xpr-2pr_4_0.xpfm
endif

KEEP_TEMP=1
KERNEL_DEBUG=
XCLBIN_NAME=kmeans
XCLBIN_HW = ${XCLBIN_NAME}_hw.xclbin
XCLBIN_SWEMU = ${XCLBIN_NAME}_sw_emu.xclbin
XCLBIN_HWEMU = ${XCLBIN_NAME}_hw_emu.xclbin

HOST_ARGS = 1

CC = g++
CLCC = xocc
CLCC_OPT += $(CLCC_OPT_LEVEL) --platform ${PLATFORM} ${KERNEL_DEFS} ${KERNEL_INCS}


ifeq (${KEEP_TEMP},1)
    CLCC_OPT += -s
endif

ifeq (${KERNEL_DEBUG},1)
    CLCC_OPT += -g
endif

ifneq (${KERNEL_FREQ},)
	CLCC_OPT += --kernel_frequency ${KERNEL_FREQ}
endif

CLCC_OPT += --kernel ${KERNEL_NAME}

.PHONY: all clean cleanall bit sim hw_sim csim run run_emu run_sw_emu run_hw_emu

all: ${HOST} ${XCLBIN_HW}

sim: 	${XCLBIN_SWEMU}	

hw_sim: ${XCLBIN_HWEMU}

# for quick verfication using GCC
csim: 
	make host DO_CSIM=1

bit: 	${XCLBIN_HW}

run: ${HOST} ${XCLBIN_HW}
	${HOST} ${XCLBIN_HW} ${HOST_ARGS}

run_csim : csim
	${HOST} dummy ${HOST_ARGS}

run_emu: run_sw_emu

run_sw_emu: emconfig.json ${HOST} ${XCLBIN_SWEMU}
	XCL_EMULATION_MODE=true ${HOST} ${XCLBIN_SWEMU} ${HOST_ARGS}

run_hw_emu: emconfig.json ${HOST} ${XCLBIN_HWEMU}
	XCL_EMULATION_MODE=true ${HOST} ${XCLBIN_HWEMU} ${HOST_ARGS}

host: ${HOST}

${HOST}: $(OBJECTS)
	$(CC) -o $@ $^ $(HOST_CFLAGS) $(LINK)  \
			-L./ \
			-Wl,-rpath=./

%.o: %.cpp
	${CC} ${HOST_CFLAGS} -c $< -o $@

hls:
	${CLCC} -c -s -t hw_emu -o kernel_top.xo --platform ${PLATFORM} --report estimate --kernel ${KERNEL_NAME} ${KERNEL_SRCS}

emconfig.json:
	emconfigutil --xdevice ${PLATFORM} ${DEVICE_REPO_OPT} --od .

${XCLBIN_SWEMU}: ${KERNEL_SRCS}
	${CLCC} -o $@ ${CLCC_OPT} -t sw_emu ${KERNEL_SRCS}

${XCLBIN_HWEMU}: ${KERNEL_SRCS}
	${CLCC} -o $@ ${CLCC_OPT} -t hw_emu ${KERNEL_SRCS} 

${XCLBIN_HW}: ${KERNEL_SRCS}
	${CLCC} -o $@ ${CLCC_OPT} -t hw ${KERNEL_SRCS}

# by default let's only clean host compile files
clean:
	${RM} -rf ${HOST} ${OBJECTS} ${XCLBIN_HWEMU} ${XCLBIN_SWEMU}

cleanall: clean
	${RM} -rf *.xclbin sdaccel_profile_summary.* _xocc_compile* _xocc_link* _xocc_krnl* TempConfig emconfig.json _xocc_*_${XCLBIN_NAME}_*.dir .Xil
