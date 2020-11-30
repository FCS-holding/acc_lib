/************************************************************************************
 Merlin Compiler (TM) Version 2020.2.dev (sprint-43-pre-278-g2b2306eb8d.14984)
 Built Mon Apr 20 04:35:55 EDT 2020
 Copyright (C) 2015-2020 Falcon Computing Solutions, Inc. All Rights Reserved.
************************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
extern int __merlin_init(const char * bitstream);
extern int __merlin_release();
extern void __merlin_conv_3d_kernel(float *data_in, float filter[24 * 24 * 24], float *data_out);
#ifdef __cplusplus
}
#endif
