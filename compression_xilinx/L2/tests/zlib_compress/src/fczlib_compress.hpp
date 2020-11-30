void fczlib_init(const char* bin_name = "kernel_top.xclbin", uint32_t file_size = 256*1024*1024);
void fczlib_compress(
   uint8_t * in,
   uint32_t * in_size,
   uint8_t * out,
   uint32_t * out_size,
   uint8_t  compress_num,//max 1024
   uint32_t compress_config
);
void fczlib_release();
