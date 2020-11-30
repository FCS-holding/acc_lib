#ifdef __cplusplus
extern "C" {
#endif
extern int __merlin_init(const char * bitstream);
extern int __merlin_release();
extern int __merlin_init_conv_3d_kernel();
extern int __merlin_release_conv_3d_kernel();
extern int __merlin_wait_kernel_conv_3d_kernel();
extern int __merlin_wait_write_conv_3d_kernel();
extern int __merlin_wait_read_conv_3d_kernel();
extern void __merlinwrapper_conv_3d_kernel(float data_in[24 * 100 * 100],float filter[24 * 24 * 24],float data_out[(100 - 24 + 1) * (100 - 24 + 1)]);
extern void __merlin_write_buffer_conv_3d_kernel(float data_in[24 * 100 * 100],float filter[24 * 24 * 24],float data_out[(100 - 24 + 1) * (100 - 24 + 1)]);
extern void __merlin_read_buffer_conv_3d_kernel(float data_in[24 * 100 * 100],float filter[24 * 24 * 24],float data_out[(100 - 24 + 1) * (100 - 24 + 1)]);
extern void __merlin_execute_conv_3d_kernel(float data_in[24 * 100 * 100],float filter[24 * 24 * 24],float data_out[(100 - 24 + 1) * (100 - 24 + 1)]);
extern void __merlin_conv_3d_kernel(float data_in[24 * 100 * 100],float filter[24 * 24 * 24],float data_out[(100 - 24 + 1) * (100 - 24 + 1)]);
#ifdef __cplusplus
}
#endif
