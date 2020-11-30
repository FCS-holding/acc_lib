#
# Copyright 2019 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# ------------------------------------------------------------
#						Help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"
	$(ECHO) "      Command to generate the design for specified Target and Device."
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""

# ------------------------------------------------------------
#						Build Environment Setup
SHELL = /bin/bash

include ./utils.mk
include ./config.mk

TOOL_VERSION ?= 2019.2

#check environment setup
ifndef XILINX_VITIS
  XILINX_VITIS = /opt/xilinx/Vitis/$(TOOL_VERSION)
  export XILINX_VITIS
endif
ifndef XILINX_VIVADO
  XILINX_VIVADO = /opt/xilinx/Vivado/$(TOOL_VERSION)
  export XILINX_VIVADO
endif
ifndef XILINX_XRT
  XILINX_XRT = /opt/xilinx/xrt
  export XILINX_XRT
endif

.PHONY: check_vivado
check_vivado:
ifeq (,$(wildcard $(XILINX_VIVADO)/bin/vivado))
	@echo "Cannot locate Vivado installation. Please set XILINX_VIVADO variable." && false
endif

.PHONY: check_vpp
check_vpp:
ifeq (,$(wildcard $(XILINX_VITIS)/bin/v++))
	@echo "Cannot locate Vitis installation. Please set XILINX_VITIS variable." && false
endif

.PHONY: check_xrt
check_xrt:
ifeq (,$(wildcard $(XILINX_XRT)/lib/libxilinxopencl.so))
	@echo "Cannot locate XRT installation. Please set XILINX_XRT variable." && false
endif

export PATH := $(XILINX_VITIS)/bin:$(XILINX_XRT)/bin:$(PATH)

ifeq (,$(LD_LIBRARY_PATH))
LD_LIBRARY_PATH := $(XILINX_XRT)/lib
else
LD_LIBRARY_PATH := $(XILINX_XRT)/lib:$(LD_LIBRARY_PATH)
endif
ifneq (,$(wildcard $(XILINX_VITIS)/bin/ldlibpath.sh))
export LD_LIBRARY_PATH := $(shell $(XILINX_VITIS)/bin/ldlibpath.sh $(XILINX_VITIS)/lib/lnx64.o):$(LD_LIBRARY_PATH)
endif

# sw_emu, hw_emu, hw
TARGET ?= sw_emu
ifeq ($(filter $(TARGET),sw_emu hw_emu hw),)
$(error TARGET is not sw_emu, hw_emu or hw)
endif

# Target device
#DEVICE ?= xilinx_u250_xdma_201830_1
#DEVICE ?= xilinx_u250_xdma_201830_2
DEVICE ?= xilinx_u50_gen3x16_xdma_201920_3

ifneq (,$(wildcard $(DEVICE)))
# Use DEVICE as a file path
XPLATFORM := $(DEVICE)
else
# Use DEVICE as a file name pattern
DEVICE_L := $(shell echo $(DEVICE) | tr A-Z a-z)
# Match the name
ifneq (,$(PLATFORM_REPO_PATHS))
XPLATFORMS := $(foreach p, $(subst :, ,$(PLATFORM_REPO_PATHS)), $(wildcard $(p)/*/*.xpfm))
XPLATFORM := $(strip $(foreach p, $(XPLATFORMS), $(shell echo $(p) | awk '$$1 ~ /$(DEVICE_L)/')))
endif
ifeq (,$(XPLATFORM))
XPLATFORMS := $(wildcard $(XILINX_VITIS)/platforms/*/*.xpfm)
XPLATFORM := $(strip $(foreach p, $(XPLATFORMS), $(shell echo $(p) | awk '$$1 ~ /$(DEVICE_L)/')))
endif
ifeq (,$(XPLATFORM))
XPLATFORMS := $(wildcard /opt/xilinx/platforms/*/*.xpfm)
XPLATFORM := $(strip $(foreach p, $(XPLATFORMS), $(shell echo $(p) | awk '$$1 ~ /$(DEVICE_L)/')))
endif
endif

define MSG_PLATFORM
No platform matched pattern '$(DEVICE)'.
Available platforms are: $(XPLATFORMS)
To add more platform directories, set the PLATFORM_REPO_PATHS variable.
endef
export MSG_PLATFORM

define MSG_DEVICE
More than one platform matched: $(XPLATFORM)
Please set DEVICE variable more accurately to select only one platform file. For example: DEVICE='u200.*xdma'
endef
export MSG_DEVICE

.PHONY: check_platform
check_platform:
ifeq (,$(XPLATFORM))
	@echo "$${MSG_PLATFORM}" && false
endif
ifneq (,$(word 2,$(XPLATFORM)))
	@echo "$${MSG_DEVICE}" && false
endif

XDEVICE := $(basename $(notdir $(firstword $(XPLATFORM))))

# ------------------------------------------------------------
#						Directory Setup

XF_PROJ_ROOT ?= $(CUR_DIR)/../../..
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))

XFLIB_DIR := $(shell readlink -f $(XF_PROJ_ROOT))

BUILD_DIR := $(CUR_DIR)/build
TEMP_DIR := $(CUR_DIR)/_x_temp.$(TARGET).$(XDEVICE)
SRC_DIR := $(XFLIB_DIR)/L2/tests/zlib_compress_sub_buffer/
TB_DIR := $(XFLIB_DIR)/L2/tests/src_sub_buffer/

# ------------------------------------------------------------
#                      kernel setup

PARALLEL_BLOCK:=8

ifeq ($(DEVICE),xilinx_u50_gen3x16_xdma_201920_3)
PLATFORM=50
else
PLATFORM=250
endif

KSRC_DIR = $(XFLIB_DIR)/L2/src/

VPP = $(XILINX_VITIS)/bin/v++
#VPP = /curr/software/Xilinx/SDx/2019.1/bin/xocc

# HLS src files
HLS_SRC_DIR = $(XFLIB_DIR)/L1/include/hw

# Compilation flags
VPP_FLAGS = -I$(HLS_SRC_DIR) \
			-I$(KSRC_DIR) \
			-I$(XFLIB_DIR)/L2/include/

VPP_FLAGS += -g -t $(TARGET) --platform $(XPLATFORM) --save-temps
VPP_FLAGS += --config $(CUR_DIR)/advanced.ini \
			 -DPARALLEL_BLOCK=$(PARALLEL_BLOCK) \
			 -DPLATFORM=$(PLATFORM)
#VPP_FLAGS += --ini_file $(CUR_DIR)/advanced.ini \
#			 -DPARALLEL_BLOCK=$(PARALLEL_BLOCK)


VPP_DIRS = --temp_dir $(TEMP_DIR)/_x.$(TARGET) \
			--report_dir $(CUR_DIR)/reports/_x.$(TARGET)

# Linking flags
VPP_LINK_FLAGS = --optimize 2 --jobs 8 \
				  --config $(CUR_DIR)/opts.ini
#VPP_LINK_FLAGS = --optimize 2 --jobs 8 \
#				  --ini_file $(CUR_DIR)/opts.ini

VPP_LINK_DIRS = --temp_dir $(TEMP_DIR)/_build.$(TARGET)\
				 --report_dir $(CUR_DIR)/reports/_build.$(TARGET)/

XCLBIN_FILE = $(BUILD_DIR)/xclbin_$(XDEVICE)_$(TARGET)/compress.xclbin

XO_FILES_C = $(TEMP_DIR)/xf_compress_lz77.xo \
		   	 $(TEMP_DIR)/xf_huffman.xo \
		  	 $(TEMP_DIR)/xf_treegen.xo \
             $(TEMP_DIR)/xf_buffer.xo
XO_FILES_C = $(TEMP_DIR)/zlib_top.xo
		  	 

COMPRESS_KERNEL_SRCS = $(KSRC_DIR)/zlib_lz77_compress_mm_new.cpp
HUFFMAN_KERNEL_SRCS = $(KSRC_DIR)/zlib_huffman_enc_mm_new.cpp
TREEGEN_KERNEL_SRCS = $(KSRC_DIR)/zlib_treegen_mm_new.cpp
BUFFER_KERNEL_SRCS = $(KSRC_DIR)/zlib_s2_pingpong_buffer.cpp
ZLIB_KERNEL_SRCS = $(KSRC_DIR)/zlib_top_loop.cpp
#ZLIB_KERNEL_SRCS = $(KSRC_DIR)/zlib_top.cpp $(COMPRESS_KERNEL_SRCS) $(HUFFMAN_KERNEL_SRCS) $(TREEGEN_KERNEL_SRCS)

COMPRESS_KERNEL_NAME = xilLz77Compress
HUFFMAN_KERNEL_NAME = xilHuffmanKernel
TREEGEN_KERNEL_NAME = xilTreegenKernel
ZLIB_KERNEL_NAME = zlib_top

KERNELS += $(COMPRESS_KERNEL_NAME) $(HUFFMAN_KERNEL_NAME) $(TREEGEN_KERNEL_NAME) 

# ------------------------------------------------------------
#                      kernel rules

# Building kernel
$(TEMP_DIR)/xf_compress_lz77.xo: $(COMPRESS_KERNEL_SRCS) $(HLS_SRC_DIR)
	@echo -e "----\nCompiling lz77 kernel $*..."
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) $(VPP_DIRS) -c -k $(COMPRESS_KERNEL_NAME) \
		-I'$(<D)' -o'$@' '$<'

$(TEMP_DIR)/xf_huffman.xo: $(HUFFMAN_KERNEL_SRCS) $(HLS_SRC_DIR)
	@echo -e "----\nCompiling huffman kernel $*..."
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) $(VPP_DIRS) -c -k $(HUFFMAN_KERNEL_NAME) \
		-I'$(<D)' -o'$@' '$<'

$(TEMP_DIR)/xf_treegen.xo: $(TREEGEN_KERNEL_SRCS) $(HLS_SRC_DIR)
	@echo -e "----\nCompiling treegen kernel $*..."
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) $(VPP_DIRS) -c -k $(TREEGEN_KERNEL_NAME) \
		-I'$(<D)' -o'$@' '$<'

$(TEMP_DIR)/zlib_top.xo: $(ZLIB_KERNEL_SRCS) $(HLS_SRC_DIR)
	@echo -e "----\nCompiling zlib_top kernel $*..."
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) $(VPP_DIRS) -c -k $(ZLIB_KERNEL_NAME) \
		-I'$(<D)' -o'$@' '$<' $(COMPRESS_KERNEL_SRCS) $(HUFFMAN_KERNEL_SRCS) $(TREEGEN_KERNEL_SRCS) $(BUFFER_KERNEL_SRCS) 

# xclbin Binary creation
$(XCLBIN_FILE): $(XO_FILES_C)
	mkdir -p $(BUILD_DIR)
	$(VPP) $(VPP_FLAGS) $(VPP_LINK_FLAGS) $(VPP_LINK_DIRS) -l -o'$@' $(+)

hls: hls1 hls2 hls3
hls1: $(COMPRESS_KERNEL_SRCS) $(HLS_SRC_DIR)
	$(VPP) $(VPP_FLAGS) --temp_dir hls1 --report_dir hls1_report -c -k $(COMPRESS_KERNEL_NAME) --report_level estimate -I'$(<D)' '$<' 
hls2: $(TREEGEN_KERNEL_SRCS) $(HLS_SRC_DIR)
	$(VPP) $(VPP_FLAGS) --temp_dir hls2 --report_dir hls2_report -c -k $(TREEGEN_KERNEL_NAME)  --report_level estimate -I'$(<D)' '$<'
hls3: $(HUFFMAN_KERNEL_SRCS) $(HLS_SRC_DIR)
	$(VPP) $(VPP_FLAGS) --temp_dir hls3 --report_dir hls3_report -c -k $(HUFFMAN_KERNEL_NAME)  --report_level estimate -I'$(<D)' '$<'
hls4: $(ZLIB_KERNEL_SRCS) $(HLS_SRC_DIR)
	echo $(ZLIB_KERNEL_SRCS)
	$(VPP) $(VPP_FLAGS) --temp_dir hls4 --report_dir hls4_report -c -k $(ZLIB_KERNEL_NAME)  --report_level estimate -I'$(<D)'  $(COMPRESS_KERNEL_SRCS) $(HUFFMAN_KERNEL_SRCS) $(TREEGEN_KERNEL_SRCS) $(BUFFER_KERNEL_SRCS) '$<'
#	$(VPP) $(VPP_FLAGS) --temp_dir hls4 --report_dir hls4_report -c -k $(ZLIB_KERNEL_NAME)  --report_level estimate -I'$(<D)'  $(HUFFMAN_KERNEL_SRCS) '$<'
#	$(VPP) $(VPP_FLAGS) --temp_dir hls4 --report_dir hls4_report -c -k $(ZLIB_KERNEL_NAME)  --report_level estimate -I'$(<D)' '$<'
#	$(VPP) $(VPP_FLAGS) --temp_dir hls4 --report_dir hls4_report -c -k $(ZLIB_KERNEL_NAME)  --report_level estimate -I'$(<D)'  $(COMPRESS_KERNEL_SRCS) $(TREEGEN_KERNEL_SRCS) '$<'
#	$(VPP) $(VPP_FLAGS) --temp_dir hls4 --report_dir hls4_report -c -k $(ZLIB_KERNEL_NAME)  --report_level estimate -I'$(<D)'  $(COMPRESS_KERNEL_SRCS) $(HUFFMAN_KERNEL_SRCS) $(TREEGEN_KERNEL_SRCS) '$<'
#	$(VPP) $(VPP_FLAGS) --temp_dir hls4 --report_dir hls4_report -c -k $(ZLIB_KERNEL_NAME)  --report_level estimate -I'$(<D)'  $(HUFFMAN_KERNEL_SRCS) '$<'
#	$(VPP) $(VPP_FLAGS) --temp_dir hls4 --report_dir hls4_report -c -k $(ZLIB_KERNEL_NAME)  --report_level estimate -I'$(<D)'  $(COMPRESS_KERNEL_SRCS) $(HUFFMAN_KERNEL_SRCS) '$<'
#	$(VPP) $(VPP_FLAGS) --temp_dir hls4 --report_dir hls4_report -c -k $(ZLIB_KERNEL_NAME)  --report_level estimate -I'$(<D)'  $(COMPRESS_KERNEL_SRCS) '$<'
#	$(VPP) $(VPP_FLAGS) --temp_dir hls4 --report_dir hls4_report -c -k $(ZLIB_KERNEL_NAME)  --report_level estimate -I'$(<D)'  $(COMPRESS_KERNEL_SRCS) $(HUFFMAN_KERNEL_SRCS) $(TREEGEN_KERNEL_SRCS) '$<'
#	$(VPP) $(VPP_FLAGS) --temp_dir hls4 --report_dir hls4_report -c -k $(ZLIB_KERNEL_NAME)  --report_level estimate -I'$(<D)' '$<'
ifeq ($(DEVICE),xilinx_u50_gen3x16_xdma_201920_3)
step_parameter= -o kernel_top.xclbin --link zlib_top_kernel.xo \
		--nk zlib_top:4:zlib_top_kernel0.zlib_top_kernel1.zlib_top_kernel2.zlib_top_kernel3 \
		--profile_kernel [data]:[all]:[all]:[all]:[all] \
		--profile_kernel [stall]:[all]:[all]:[all]:[all] \
		--profile_kernel [exec]:[all]:[all]:[all]:[all] \
		--slr zlib_top_kernel0:SLR0 \
		--slr zlib_top_kernel1:SLR0 \
		--slr zlib_top_kernel2:SLR1 \
		--slr zlib_top_kernel3:SLR1 \
		--sp zlib_top_kernel0.m_axi_gmem0:HBM[0] \
		--sp zlib_top_kernel0.m_axi_gmem1:HBM[1] \
		--sp zlib_top_kernel0.m_axi_gmem2:HBM[2] \
		--sp zlib_top_kernel0.m_axi_gmem3:HBM[3] \
		--sp zlib_top_kernel1.m_axi_gmem0:HBM[4] \
		--sp zlib_top_kernel1.m_axi_gmem1:HBM[5] \
		--sp zlib_top_kernel1.m_axi_gmem2:HBM[6] \
		--sp zlib_top_kernel1.m_axi_gmem3:HBM[7] \
		--sp zlib_top_kernel2.m_axi_gmem0:HBM[8] \
		--sp zlib_top_kernel2.m_axi_gmem1:HBM[9] \
		--sp zlib_top_kernel2.m_axi_gmem2:HBM[10] \
		--sp zlib_top_kernel2.m_axi_gmem3:HBM[11] \
		--sp zlib_top_kernel3.m_axi_gmem0:HBM[12] \
		--sp zlib_top_kernel3.m_axi_gmem1:HBM[13] \
		--sp zlib_top_kernel3.m_axi_gmem2:HBM[14] \
		--sp zlib_top_kernel3.m_axi_gmem3:HBM[15] \

else

step_parameter= -o kernel_top.xclbin --link zlib_top_kernel.xo \
		--nk zlib_top:4:zlib_top_kernel0.zlib_top_kernel1.zlib_top_kernel2.zlib_top_kernel3 \
		--profile_kernel [data]:[all]:[all]:[all]:[all] \
		--profile_kernel [stall]:[all]:[all]:[all]:[all] \
		--profile_kernel [exec]:[all]:[all]:[all]:[all] \
		--slr zlib_top_kernel0:SLR0 \
		--slr zlib_top_kernel1:SLR1 \
		--slr zlib_top_kernel2:SLR2 \
		--slr zlib_top_kernel3:SLR3 \
		--sp zlib_top_kernel0.m_axi_gmem0:DDR[0] \
		--sp zlib_top_kernel0.m_axi_gmem1:DDR[0] \
		--sp zlib_top_kernel1.m_axi_gmem0:DDR[1] \
		--sp zlib_top_kernel1.m_axi_gmem1:DDR[1] \
		--sp zlib_top_kernel2.m_axi_gmem0:DDR[2] \
		--sp zlib_top_kernel2.m_axi_gmem1:DDR[2] \
		--sp zlib_top_kernel3.m_axi_gmem0:DDR[3] \
		--sp zlib_top_kernel3.m_axi_gmem1:DDR[3] \

endif


step: step1 step2

step1:  $(ZLIB_KERNEL_SRCS) $(HLS_SRC_DIR)
	$(VPP)  $(VPP_FLAGS) --temp_dir step --report_dir step_report -o zlib_top_kernel.xo \
		-c -k $(ZLIB_KERNEL_NAME)  --profile_kernel [stall]:[all]:[all]:[all]:[all] -I'$(<D)'  $(COMPRESS_KERNEL_SRCS) $(HUFFMAN_KERNEL_SRCS) $(TREEGEN_KERNEL_SRCS) $(BUFFER_KERNEL_SRCS) '$<'
	#$(VPP) $(VPP_FLAGS) -o zlib_top_kernel.xo -c $(ZLIB_KERNEL_SRCS) -k $(ZLIB_KERNEL_NAME) --report_level estimate -I'$(<D)'  $(COMPRESS_KERNEL_SRCS) $(HUFFMAN_KERNEL_SRCS) $(TREEGEN_KERNEL_SRCS) $(BUFFER_KERNEL_SRCS) '$<'
	#xocc -t hw -o zlib_top_kernel.xo -c zlib_top.cpp -k zlib_top_kernel -s --platform xilinx_u250_xdma_201830_1
	#xocc -t sw_emu -o zlib_top_kernel.xo -c zlib_top.cpp -k zlib_top_kernel -s --platform xilinx_u250_xdma_201830_1

step2: $(ZLIB_KERNEL_SRCS) $(HLS_SRC_DIR)
	$(VPP) $(VPP_FLAGS) $(step_parameter)

#step2: $(ZLIB_KERNEL_SRCS) $(HLS_SRC_DIR)
#	$(VPP) $(VPP_FLAGS) -o kernel_top.xclbin --link zlib_top_kernel.xo \
#		--nk zlib_top:4:zlib_top_kernel0.zlib_top_kernel1.zlib_top_kernel2.zlib_top_kernel3 \
#		--profile_kernel [data]:[all]:[all]:[all]:[all] \
#		--profile_kernel [stall]:[all]:[all]:[all]:[all] \
#		--profile_kernel [exec]:[all]:[all]:[all]:[all] \
#		--slr zlib_top_kernel0:SLR0 \
#		--slr zlib_top_kernel1:SLR1 \
#		--slr zlib_top_kernel2:SLR2 \
#		--slr zlib_top_kernel3:SLR3 \
#		--sp zlib_top_kernel0.m_axi_gmem0:DDR[0] \
#		--sp zlib_top_kernel0.m_axi_gmem1:DDR[0] \
#		--sp zlib_top_kernel1.m_axi_gmem0:DDR[1] \
#		--sp zlib_top_kernel1.m_axi_gmem1:DDR[1] \
#		--sp zlib_top_kernel2.m_axi_gmem0:DDR[2] \
#		--sp zlib_top_kernel2.m_axi_gmem1:DDR[2] \
#		--sp zlib_top_kernel3.m_axi_gmem0:DDR[3] \
#		--sp zlib_top_kernel3.m_axi_gmem1:DDR[3] \
#
# ------------------------------------------------------------
#                       host setup

#CXX := xcpp
CXX := g++
HOST_EXE := xil_zlib
#EXE_EXT = exe

CXXFLAGS +=-I$(CUR_DIR)/src/
CXXFLAGS +=-I$(XFLIB_DIR)/L2/include/
CXXFLAGS +=-I$(TB_DIR)/
CXXFLAGS +=-I$(XILINX_XRT)/include/
CXXFLAGS +=-I$(XFLIB_DIR)/common/libs/xcl2/
CXXFLAGS +=-I$(XFLIB_DIR)/common/libs/cmdparser/
CXXFLAGS +=-I$(XFLIB_DIR)/common/libs/logger/
CXXFLAGS +=-I$(XFLIB_DIR)/common/thirdParty/xxhash/

#Host and Common sources
SRCS += $(SRC_DIR)/src/host.cpp
#SRCS += $(SRC_DIR)/src/fczlib_compress.cpp
SRCS += $(TB_DIR)/zlib.cpp
SRCS += $(XFLIB_DIR)/common/libs/xcl2/xcl2.cpp
SRCS += $(XFLIB_DIR)/common/libs/cmdparser/cmdlineparser.cpp
SRCS += $(XFLIB_DIR)/common/libs/logger/logger.cpp
SRCS += $(XFLIB_DIR)/common/thirdParty/xxhash/xxhash.c

CXXFLAGS += -fmessage-length=0 -std=c++11 -O3 \
		-DXDEVICE=$(XDEVICE) \
		-DPLATFORM=$(PLATFORM) \
	    -Wall -Wno-unknown-pragmas -Wno-unused-label -pthread

LDFLAGS += -L$(XILINX_XRT)/lib/ -lxilinxopencl -pthread 
#-L. -lkernel
#LDFLAGS += -L$(XILINX_XRT)/lib/ -lOpenCL -pthread
#LDFLAGS += -lrt -Wno-unused-label -Wno-narrowing -std=c++14 -DVERBOSE -lstdc++ -std=gnu++0x -DLINUX_COMMANDS -fPIC -DIS_PIC -ldl
#LDFLAGS += -lrt -Wno-unused-label -Wno-narrowing -std=c++14 -DVERBOSE -static-libstdc++
LDFLAGS += -lrt -Wno-unused-label -Wno-narrowing -std=c++11 -DVERBOSE -lstdc++

#LICFLAGS += $(XFLIB_DIR)/license/license.o $(XFLIB_DIR)/license/lm_new.o
#LICFLAGS += -I $(XFLIB_DIR)/license/ 
#LICFLAGS += -L $(XFLIB_DIR)/license/ -Wl,-rpath=$(XFLIB_DIR)/license/
#LICFLAGS += -llmgr_trl -lcrvs -lsb -lnoact -llmgr_dongle_stub -lpthread -ldl

EXE_FILE = $(BUILD_DIR)/$(HOST_EXE)
CC = gcc

#-----------------------------------------------------------
#	lib_gen
lib_gen:
	@rm -rf bin;
	@mkdir bin;
	$(ECHO)	$(CXX) -O -shared -fPIC -DHLS_NO_XIL_FPO_LIB -o ./libkernel.so  $(SRCS)  $(CXXFLAGS) $(LDFLAGS) $(LICFLAGS) 


# ------------------------------------------------------------
#                       host rules

$(EXE_FILE): $(SRCS) | check_vpp check_xrt check_platform
	@echo -e "----\nCompiling host $(notdir $@)..."
	mkdir -p $(BUILD_DIR)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) $(LICFLAGS)


# ------------------------------------------------------------
#                      build rules

.PHONY: all help host xclbin cleanh cleank cleanall clean

all: host xclbin

host: $(EXE_FILE) | check_vpp check_xrt check_platform

xclbin: $(XCLBIN_FILE)

cleank:
	rm -f _x_temp.*/*.xo
	rm -f $(BUILD_DIR)/*.xclbin
	rm -rf _x_temp.*/_x.* _x_temp.*/.Xil _x_temp.*/profile_summary.* sample_*
	rm -rf _x_temp.*/dltmp* _x_temp.*/kernel_info.dat _x_temp.*/*.log
	rm -rf _x_temp.*
	
cleanh:
	rm -rf $(EXE_FILE)
	-$(RMDIR) $(EXE_FILE)
	-$(RMDIR) vitis_* TempConfig system_estimate.xtxt *.rpt .run/
	-$(RMDIR) src/*.ll _xocc_* .Xil dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: cleanh cleank
	rm -rf $(BUILD_DIR)
	-$(RMDIR) $(TEMP_DIR) $(CUR_DIR)/reports *.csv $(CUR_DIR)/*.raw
	-$(RMDIR) $(XFLIB_DIR)/common/data/*.xe2xd* $(XFLIB_DIR)/common/data/*.orig*

clean: cleanh


# ------------------------------------------------------------
#                      simulation run

$(BUILD_DIR)/emconfig.json :
		emconfigutil --platform $(XPLATFORM) --od $(BUILD_DIR)

HOST_ARGS = -cx $(XCLBIN_FILE) -c $(XFLIB_DIR)/common/data/sample.txt
ifeq ($(TARGET),sw_emu)
  RUN_ENV = export XCL_EMULATION_MODE=sw_emu
  EMU_CONFIG = $(BUILD_DIR)/emconfig.json
else ifeq ($(TARGET),hw_emu)
  RUN_ENV = export XCL_EMULATION_MODE=hw_emu
  EMU_CONFIG = $(BUILD_DIR)/emconfig.json
else ifeq ($(TARGET),hw)
  RUN_ENV = echo "TARGET=hw"
  EMU_CONFIG =
endif


run: host xclbin $(EMU_CONFIG) $(DAT_STAMP)
	$(RUN_ENV); \
	$(EXE_FILE) $(HOST_ARGS)

check: run

.PHONY: build
build: xclbin host
