
#ifndef __OPENCL_IF_H_INCLUDED__
#define __OPENCL_IF_H_INCLUDED__
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <math.h>
#include <time.h>
#include <string>
#include <fstream>
#include <thread>
#include <stdlib.h>
#include "__merlin_parameter.h"
#include "xcl2.hpp"
#include <CL/opencl.h>
#include <CL/cl2.hpp>
#include <CL/cl_ext.h>
#define __MERLIN_SDACCEL
#ifdef __MERLIN_SDACCEL
#include <CL/cl_ext_xilinx.h>
#endif

//template <typename T>
//struct aligned_allocator {
//    using value_type = T;
//    T* allocate(std::size_t num) {
//        void* ptr = nullptr;
//        if (posix_memalign(&ptr, 4096, num * sizeof(T))) throw std::bad_alloc();
//        return reinterpret_cast<T*>(ptr);
//    }
//    void deallocate(T* p, std::size_t num) { free(p); }
//};

#ifdef __cplusplus
extern "C" {
#endif
int opencl_init(const char *bitstream);
int __merlin_init(const char *bitstream);
int __merlin_init_opencl(const char *bitstream);
int __merlin_release();
int __merlin_release_opencl();
int __merlin_check_opencl();
#ifdef __cplusplus
}
#endif
// int opencl_create_kernel (char * kernel_name);
int opencl_create_kernel(cl_kernel *kernel, char *kernel_name);
int opencl_create_buffer(cl_mem *cl_buffer, long long size, int type);
#ifdef __MERLIN_SDACCEL
int opencl_create_ext_buffer(cl_mem *cl_buffer, cl_mem_ext_ptr_t *cl_buffer_ext,
                             long long size, int type);
#endif
int opencl_write_buffer(cl_mem cl_buffer, cl_command_queue commands,
                        long long offset, void *host_buffer, long long size);
int opencl_read_buffer(cl_mem cl_buffer, cl_command_queue commands,
                       long long offset, void *host_buffer, long long size);
int opencl_write_buffer_nb(cl_mem cl_buffer, cl_command_queue commands,
                           long long offset, void *host_buffer, long long size,
                           cl_event *event_in, cl_event *event);
int opencl_read_buffer_nb(cl_mem cl_buffer, cl_command_queue commands,
                          long long offset, void *host_buffer, long long size,
                          cl_event *event_in, cl_event *event);
int opencl_set_kernel_arg(cl_kernel kernel, int index, size_t size,
                          const void *content);
// int opencl_enqueue_kernel(cl_kernel kernel, int dim, size_t global[]);
int opencl_enqueue_kernel(cl_kernel kernel, cl_command_queue commands, int dim,
                          size_t global_in[], cl_event *event_in, cl_event *event_out);
int opencl_enqueue_task(cl_kernel kernel, cl_command_queue commands);
int opencl_flush(cl_command_queue commands);

int opencl_wait_event(cl_event event);
int opencl_release_event(cl_event event);
int opencl_release_mem_object(cl_mem mem);
int opencl_release_kernel(cl_kernel kernel);

extern cl_command_queue commands[QUEUE_NUM];

typedef cl_mem opencl_mem;
typedef cl_kernel opencl_kernel;
#ifdef __MERLIN_SDACCEL
typedef cl_mem_ext_ptr_t opencl_mem_ext;
#endif

#define COMPUTE_UNIT 1
extern cl::CommandQueue* m_q[OVERLAP*PE]; 
extern cl::Buffer* buffer_filter[PE];
extern cl::Buffer* buffer_input[PE][OVERLAP];
extern cl::Buffer* buffer_output[PE][OVERLAP];
extern cl::Kernel* conv_kernel[PE];

extern std::vector<float, aligned_allocator<float> > input_align[PE][OVERLAP];  
extern std::vector<float, aligned_allocator<float> > output_align[PE][OVERLAP];        
extern std::vector<float, aligned_allocator<float> > filter_align[PE];  
#endif //__OPENCL_IF_H_INCLUDED__
