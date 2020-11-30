PP=g++
CC=gcc

CFLAGS= -c -fPIC -std=c++0x -m64 -g -DBLAZE_TEST
COMPILE= $(CFLAGS) \
				 -I$(CUDA_PATH)/include

LINK= -L$(CUDA_PATH)/lib64 -lOpenCL \
			-lpthread -lm

all: host

host: host.o 
	$(PP) $(LINK) -o $@ host.o

%.o: %.cpp Makefile
	$(PP) $(COMPILE) $< -o $@

clean:
	rm -rf host
	rm -rf *.o
