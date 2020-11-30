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
			-lboost_system -lboost_thread -lboost_iostreams -lboost_filesystem \
			-L$(PROTOBUF_DIR)/lib -lprotobuf \
			-lpthread -lm -ldl

all: run_test LogisticGradientWithLoss.so

run_test: LogisticGradient.o
	$(PP) -o $@ LogisticGradient.o $(LINK) 

LogisticGradientWithLoss.so: LogisticGradientWithLoss.o gradient.o
	$(PP) -shared -o $@ LogisticGradientWithLoss.o gradient.o $(LINK) 

%.o: %.cpp Makefile
	$(PP) -c $(COMPILE) $< -o $@

clean:
	rm -rf run_test
	rm -rf *.o
