#include "__merlin_opencl_if.h"
int opencl_init_kernel_buffer();
int __merlin_release_kernel_buffer();
extern opencl_kernel __conv_3d_kernel_kernel;
extern cl_event __event_conv_3d_kernel;
#ifdef __cplusplus
extern "C" {
#endif
int __merlin_init_conv_3d_kernel();
int __merlin_release_conv_3d_kernel();
int __merlin_wait_kernel_conv_3d_kernel();
int __merlin_wait_write_conv_3d_kernel();
int __merlin_wait_read_conv_3d_kernel();
#ifdef __cplusplus
}
#endif
extern opencl_mem __conv_3d_kernel__data_in_buffer;
extern cl_event __event___conv_3d_kernel__data_in_buffer;
extern opencl_mem __conv_3d_kernel__data_out_buffer;
extern cl_event __event___conv_3d_kernel__data_out_buffer;
extern opencl_mem __conv_3d_kernel__filter_buffer;
extern cl_event __event___conv_3d_kernel__filter_buffer;
