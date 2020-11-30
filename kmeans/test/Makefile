ifeq ($(BLAZE_HOME),)
$(error BLAZE_HOME not set)
endif
ifeq ("$(wildcard $(BLAZE_HOME)/Makefile.config)","")
$(error BLAZE_HOME not set properly in environment)
endif

MANAGER_DIR=$(BLAZE_HOME)/manager
PLATFORM_DIR=$(BLAZE_HOME)/platforms/xil_ocl
include $(BLAZE_HOME)/Makefile.config

PP=g++
CC=gcc

CFLAGS= -fPIC -std=c++0x -m64 -g 
COMPILE= $(CFLAGS) \
				 -I$(MANAGER_DIR)/include \
				 -I$(BOOST_DIR)/include \
				 -I$(PROTOBUF_DIR)/include \
				 -DHLS_SIM -DCPP_SIM

LINK= -L$(MANAGER_DIR)/lib -lblaze \
			-L$(BOOST_DIR)/lib \
			-lboost_system -lboost_thread -lboost_iostreams -lboost_filesystem -lboost_regex \
			-L$(PROTOBUF_DIR)/lib -lprotobuf \
			-L$(GLOG_DIR)/lib -lglog \
			-lpthread -lm -ldl

all: run_test KMeansContrib.so

run_test: host.o
	$(PP) -o $@ host.o $(LINK) 

KMeansContrib.so: KMeansContrib.o kmeans.o
	$(PP) -shared -o $@ KMeansContrib.o kmeans.o

%.o: %.cpp Makefile
	$(PP) -c $(COMPILE) $< -o $@

clean:
	rm -rf run_test
	rm -rf *.o
