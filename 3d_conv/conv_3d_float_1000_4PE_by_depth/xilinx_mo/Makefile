# Choose target FPGA platform vendor
VENDOR=XILINX
#DEVICE=xilinx_vcu1525_dynamic_5_1
DEVICE=xilinx_u250_xdma_201830_1
#VENDOR=INTEL

# Host Code Compilation settings
MACH_COMMON_DIR=../../../common
HOST_SRC_FILES=../src/host.cpp

# Executable names and arguments
EXE=conv_3d_app
ACC_EXE=$(EXE)_acc
EXE_ARGS=

CXX=g++ -I$(XILINX_XRT)/include -L$(XILINX_XRT)/lib -lxilinxopencl -L/lib64 -lxml2
CXX_INC_DIRS=-I ../src -I ./
CXX_FLAGS+= $(CXX_INC_DIRS) -Wall -O3
#CXX_FLAGS+= -Wall
#CXX_FLAGS+= -g
ifeq ($(VENDOR),XILINX)

CXX_FLAGS +=-lstdc++ -L$(XILINX_SDX)/lib/lnx64.o
endif

# Accelerated Kernel settings
KERNEL_NAME=krnl_conv_3d
KERNEL_SRC_FILES=../src/krnl_conv_3d.cpp
KERNEL_INC_DIR=$(CXX_INC_DIRS)
#CMP_OPT=-d11
CMP_OPT=-d11 --attribute memory_coalescing=off --attribute memory_burst=off -DMCC_ACC
LNK_OPT=-d11

AOC_KERNEL_SRC_FILES=../src/krnl_conv_3d.cpp
XOC_KERNEL_SRC_FILES=../src/krnl_conv_3d.cpp

MCC_COMMON_DIR=.
include $(MCC_COMMON_DIR)/mcc_common.mk
