/*******************************************************************************
 Merlin Compiler (TM) Version 1.0.0
 Build ddaed33.1417 on Tue Aug 15 17:22:56 2017 -0700
 Copyright (C) 2015-2017 Falcon Computing Solutions, Inc. All Rights Reserved.
*******************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
extern int opencl_init(const char * bitstream);extern int __merlin_init(const char * bitstream);
extern
void __merlinwrapper_kmeans_kernel(int num_samples,int num_runs,int num_clusters,int vector_length,double *data,double *centers,double *output,int data_size,int center_size,int output_size);
#ifdef __cplusplus
}
#endif
