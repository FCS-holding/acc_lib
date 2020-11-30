#include "__merlin_opencl_kernel_buffer.h"
opencl_kernel __conv_3d_kernel_kernel;
cl_event __event_conv_3d_kernel;
opencl_mem __conv_3d_kernel__data_in_buffer;
cl_event __event___conv_3d_kernel__data_in_buffer;
opencl_mem __conv_3d_kernel__data_out_buffer;
cl_event __event___conv_3d_kernel__data_out_buffer;
opencl_mem __conv_3d_kernel__filter_buffer;
cl_event __event___conv_3d_kernel__filter_buffer;
int __merlin_init_conv_3d_kernel(){
    #if VERSION==0
	printf("[Merlin Info] Start create kernel for %s....\n", "conv_3d_kernel");
	fflush(stdout);
	opencl_create_kernel(&__conv_3d_kernel_kernel, (char *)"conv_3d_kernel");
	printf("[Merlin Info] Successful create kernel for %s....\n", "conv_3d_kernel");
	fflush(stdout);
	printf("[Merlin Info] Start create buffer for %s....\n", "data_in");
	fflush(stdout);
	opencl_create_buffer(&__conv_3d_kernel__data_in_buffer, (int64_t)4*DATA_IN_LENGTH, 2);
	printf("[Merlin Info] Successful create buffer for %s....\n", "data_in");
	fflush(stdout);
	printf("[Merlin Info] Start create buffer for %s....\n", "data_out");
	fflush(stdout);
	opencl_create_buffer(&__conv_3d_kernel__data_out_buffer, (int64_t)4*DATA_OUT_LENGTH, 2);
	printf("[Merlin Info] Successful create buffer for %s....\n", "data_out");
	fflush(stdout);
	printf("[Merlin Info] Start create buffer for %s....\n", "filter");
	fflush(stdout);
	opencl_create_buffer(&__conv_3d_kernel__filter_buffer, (int64_t)4*FILTER_IN_LENGTH, 2);
	printf("[Merlin Info] Successful create buffer for %s....\n", "filter");
	fflush(stdout);
    #else
    #endif
	return 0;
}
int opencl_init_kernel_buffer(){
	int __merlin_init_conv_3d_kernel();
	return 0;
}
int __merlin_release_conv_3d_kernel(){
    #if VERSION==0
	if(__conv_3d_kernel_kernel) {
		opencl_release_kernel(__conv_3d_kernel_kernel);
	}
	if(__event_conv_3d_kernel) {
		opencl_release_event(__event_conv_3d_kernel);
	}
	if(__conv_3d_kernel__data_in_buffer) {
		opencl_release_mem_object(__conv_3d_kernel__data_in_buffer);
	}
	if(__conv_3d_kernel__data_out_buffer) {
		opencl_release_mem_object(__conv_3d_kernel__data_out_buffer);
	}
	if(__conv_3d_kernel__filter_buffer) {
		opencl_release_mem_object(__conv_3d_kernel__filter_buffer);
	}
    #else
    #endif
	return 0;
}
int __merlin_release_kernel_buffer(){
	int __merlin_release_conv_3d_kernel();
	return 0;
}
int __merlin_wait_kernel_conv_3d_kernel(){
	opencl_wait_event(__event_conv_3d_kernel);
	return 0;
}
int __merlin_wait_write_conv_3d_kernel(){
		opencl_wait_event(__event___conv_3d_kernel__data_in_buffer);
	if(__conv_3d_kernel__data_in_buffer) {
		opencl_release_mem_object(__conv_3d_kernel__data_in_buffer);
	}
		opencl_wait_event(__event___conv_3d_kernel__filter_buffer);
	if(__conv_3d_kernel__filter_buffer) {
		opencl_release_mem_object(__conv_3d_kernel__filter_buffer);
	}
	return 0;
}
int __merlin_wait_read_conv_3d_kernel(){
		opencl_wait_event(__event___conv_3d_kernel__data_out_buffer);
	if(__conv_3d_kernel__data_out_buffer) {
		opencl_release_mem_object(__conv_3d_kernel__data_out_buffer);
	}
	return 0;
}
