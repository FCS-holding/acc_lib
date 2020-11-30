#include <stdio.h>
#include <hls_stream.h>
#include "conv_3d.h"


void fadd_parallel(float *in, float *out) {
    float tmp=0;                             
    for(int i = 0; i < 24; i++){               
        tmp += in[i];                          
    }                                          
    *out = tmp;                                
}
void compute_one_plan_sub(float input[filter_h][image_width], 
                          hls::stream<float> &stream_data_in, 
                          float filter[filter_w][filter_h],
                          hls::stream<float> &stream_data_out) {
    float shift[filter_w][filter_h];

    #pragma ACCEL parallel complete
    for (int w = 0; w < filter_w; w++) {
        for (int h = 0; h < filter_h; h++) {
            shift[w][h] = ((float )0);
        }
    }

    for (int t = 0; t < STEP; t++) {
        #pragma ACCEL pipeline flatten
        for (int i = 0; i < image_width; i++) {
            float input_tmp[filter_h];
            for (int j = 0; j < filter_h; j++) {
                input_tmp[j] = input[j][i];
            }

            for (int w = filter_w-1; w > 0; w--) {
                for (int h = 0; h < filter_h; h++) {
                    shift[w][h] = shift[w-1][h];
                }
            }

            for (int h = 0; h < filter_h-1; h++) {
                input[h][i] = input[h+1][i];
            }

            input[filter_h-1][i] = stream_data_in.read();

            for (int h = 0; h < filter_h; h++) {
                shift[0][h] = input_tmp[h];
            }

            if (i >= filter_h - 1) {
                
                float tmp2[filter_w];
                float result_tmp;
                for (int w = 0; w < filter_w; w++) {
                    float tmp1[filter_h];
                    for (int h = 0; h < filter_h; h++) {
                        #pragma HLS RESOURCE variable=result_tmp core=FMul_fulldsp
                        result_tmp = shift[filter_w - w - 1][h] * filter[w][h];
                        tmp1[h] = result_tmp;
                    }
                    fadd_parallel(tmp1,&tmp2[w]);
                }
                float tmp3;
                fadd_parallel(tmp2,&tmp3);
                stream_data_out.write(tmp3);
            }
        }
    }
}

void shift_in_data(float *data_in,
                   hls::stream<float> &stream_data_in,
                   int s_p) {

    STREAM1:
    for (int i = 0; i < STEP; ++i) {
        float buf[image_width];
        STREAM2:
        for (int j = 0; j < image_width; ++j) {
            buf[j] = data_in[(s_p)*image_width*IN_LENGTH_0+image_width*filter_h+i*image_width+j];
            stream_data_in.write(buf[j]);
        }
    }
}

void adder_out(float plane_out[STEP*(image_width-filter_w+1)],
               hls::stream<float> &stream_data_out) {
    for (int i = 0; i < STEP*(image_width-filter_w+1); i++) {
        plane_out[i] += stream_data_out.read();
    }
}

void compute_one_plan(float data_in_tmp[filter_h][image_width],
                      float *data_in, 
                      float filter[filter_w][filter_h],
                      float plane_out[STEP*(image_width-filter_w+1)], 
                      int s_p) {
    #pragma HLS inline off
    #pragma HLS dataflow       

    hls::stream<float> stream_data_in;
    hls::stream<float> stream_data_out;                      

    shift_in_data(data_in, stream_data_in, s_p);                      
    compute_one_plan_sub(data_in_tmp, stream_data_in, filter, stream_data_out);
    adder_out(plane_out, stream_data_out);                      
}

void memory_burst_filter(float filter_tmp[filter_d][filter_w][filter_h],float *filter) {
    for (int i = 0; i < filter_d; i++) {
        for (int j = 0; j < filter_w; j++) {
            #pragma ACCEL pipeline flatten
            for (int k = 0; k < filter_h; k++) {
                filter_tmp[i][j][k] = filter[i*filter_w*filter_h+k*filter_h+j];
            }
        }
    }
}

void memory_burst_in(float data_in_tmp[filter_h][image_width],float *data_in, int s_p) {
    for (int j = 0; j < filter_h; j++) {
        #pragma ACCEL pipeline flatten
        for (int i = 0; i < image_width; i++) {
            data_in_tmp[j][i] = data_in[s_p * image_width * IN_LENGTH_0 + j * image_width + i];
        }
    }
}

void memory_burst_out(float plane_out[STEP*(image_width-filter_w+1)],float *merlin_output, int p) {
    for (int i = 0; i < STEP*(image_width-filter_w+1); i++) {
        merlin_output[p*STEP*(image_width-filter_w+1)+i] = plane_out[i];
    }
}

void init_plane_out(float plane_out[STEP*(image_width-filter_w+1)]){
    for (int i = 0; i < STEP*(image_width-filter_w+1); i++) {
        plane_out[i] = 0;
    }
}

void compute_cube(float *data_in, float *filter, float * data_out) {
    float filter_tmp[filter_w][filter_d][filter_h];
    #pragma HLS RESOURCE variable=filter_tmp core=RAM_2P_BRAM
    memory_burst_filter(filter_tmp, filter);
    for (int p = 0; p < image_depth-filter_d+1; p++) {
        float plane_out[STEP*(image_width-filter_w+1)];
        init_plane_out(plane_out);
        for (int s = 0; s < filter_d; s++) {
            int s_p = s + p;
            float data_in_tmp[filter_h][image_width];
            #pragma HLS RESOURCE variable=plane_out core=RAM_2P_URAM
            memory_burst_in(data_in_tmp, data_in, s_p);
            compute_one_plan(data_in_tmp, data_in, filter_tmp[s], plane_out, s_p);
        }
        memory_burst_out(plane_out, data_out, p);
    }
}

#pragma ACCEL kernel
void conv_3d_kernel0(float data_in[IN_SIZE_0], float filter[FILTER_LENGTH], float data_out[OUT_SIZE_0]){
    #pragma ACCEL interface variable=data_in  depth=KERNEL_IN_LENGTH    bank=0
    #pragma ACCEL interface variable=filter   depth=FILTER_LENGTH       bank=0
    #pragma ACCEL interface variable=data_out depth=KERNEL_OUT_LENGTH   bank=0

    compute_cube(data_in, filter, data_out);
}

#if PE>=2
#pragma ACCEL kernel
void conv_3d_kernel1(float data_in[IN_SIZE_1], float filter[FILTER_LENGTH], float data_out[OUT_SIZE_1]){
    #pragma ACCEL interface variable=data_in  depth= KERNEL_IN_LENGTH   bank=1  
    #pragma ACCEL interface variable=filter   depth= FILTER_LENGTH      bank=1
    #pragma ACCEL interface variable=data_out depth= KERNEL_OUT_LENGTH  bank=1
    
    compute_cube(data_in, filter, data_out);
}
#endif

#if PE>=4
#pragma ACCEL kernel
void conv_3d_kernel2(float data_in[IN_SIZE_2], float filter[FILTER_LENGTH], float data_out[OUT_SIZE_2]){
    #pragma ACCEL interface variable=data_in  depth= KERNEL_IN_LENGTH   bank=2 
    #pragma ACCEL interface variable=filter   depth= FILTER_LENGTH      bank=2
    #pragma ACCEL interface variable=data_out depth= KERNEL_OUT_LENGTH  bank=2
    
    compute_cube(data_in, filter, data_out);
}
#endif

#if PE>=4
#pragma ACCEL kernel
void conv_3d_kernel3(float data_in[IN_SIZE_3], float filter[FILTER_LENGTH], float data_out[OUT_SIZE_3]){
    #pragma ACCEL interface variable=data_in  depth= KERNEL_IN_LENGTH   bank=3 
    #pragma ACCEL interface variable=filter   depth= FILTER_LENGTH      bank=3
    #pragma ACCEL interface variable=data_out depth= KERNEL_OUT_LENGTH  bank=3

    compute_cube(data_in, filter, data_out);
}
#endif

#pragma ACCEL task parallel
void new_task(
    float data_in_0[IN_SIZE_0],
    #if PE>=2
    float data_in_1[IN_SIZE_1],
    #endif
    
    #if PE>=4
    float data_in_2[IN_SIZE_2],
    float data_in_3[IN_SIZE_3],
    #endif
    float data_out_0[OUT_SIZE_0],
    #if PE>=2
    float data_out_1[OUT_SIZE_1],
    #endif
    #if PE>=4
    float data_out_2[OUT_SIZE_2],
    float data_out_3[OUT_SIZE_3],
    #endif
    float filter_0[FILTER_LENGTH],
    #if PE>=2
    float filter_1[FILTER_LENGTH],
    #endif
    #if PE>=4
    float filter_2[FILTER_LENGTH],
    float filter_3[FILTER_LENGTH]
    #endif
    ){
    conv_3d_kernel0(data_in_0, filter_0, data_out_0);
    #if PE>=2
    conv_3d_kernel1(data_in_1, filter_1, data_out_1);
    #endif
    #if PE>=4
    conv_3d_kernel2(data_in_2, filter_2, data_out_2);
    conv_3d_kernel3(data_in_3, filter_3, data_out_3);
    #endif
}
