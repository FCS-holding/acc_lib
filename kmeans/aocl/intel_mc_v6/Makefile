# Choose target FPGA platform vendor
#VENDOR=XILINX
VENDOR=INTEL

# Host Code Compilation settings
#HOST_SRC_FILES=../src/host.cpp ../src/kmeans.cpp
HOST_SRC_FILES=../src/main.cpp
HOST_SRC_FILES+= ../src/kmeans_kernel_v6.cpp
#HOST_SRC_FILES+= ../src/kmeans.cpp
#HOST_SRC_FILES+= ../src/kmeans_mod.cpp

# Executable names and arguments
EXE=test_kmeans
ACC_EXE=test_acckmeans
EXE_ARGS=. 1000

CXX_INC_DIRS=-I ../include -I ./
CXX_FLAGS+= $(CXX_INC_DIRS)
CXX_FLAGS+= -Wall -O3
#CXX_FLAGS+= -g
CXX_FLAGS += -lpthread -lm -ldl -lrt -lstdc++ -std=c++0x -fPIC

# Accelerated Kernel settings
KERNEL_NAME=kmeans_kernel
KERNEL_SRC_FILES=../src/kmeans_kernel_v6.cpp
#KERNEL_SRC_FILES=../src/kmeans.cpp
#KERNEL_SRC_FILES=../src/kmeans_mod.cpp
KERNEL_INC_DIR=$(CXX_INC_DIRS)
CMP_OPT=-d11
LNK_OPT=-d11 -g --profile

MCC_COMMON_DIR=../../../common
include $(MCC_COMMON_DIR)/mcc_common.mk
