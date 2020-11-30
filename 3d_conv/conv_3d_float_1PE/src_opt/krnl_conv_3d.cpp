#include <stdio.h>
#include <hls_stream.h>
#include "conv_3d.h"

void compute_one_plan_sub(float *data_in, 
                          float *filter,
                          hls::stream<float> &stream_data_out,
                          int s,
                          int p) {
    int i, w, h;
    for (int t = 0; t < image_height; t++) {
        #pragma ACCEL line_buffer variable=data_in
        for (i = 0; i < image_width; i++) {
            if (i >= filter_h - 1 && t >= filter_w - 1) {
                float result_tmp = 0;
                for (h = 0; h < filter_h; h++) {
                    for (w = 0; w < filter_w; w++) {
                        float tmp = data_in[(s+p)*image_width*image_height+(t-filter_h+1+h)*image_width+(i-filter_w+1+w)];
                        result_tmp += tmp *  filter[s*filter_w*filter_h+h*filter_h+w];
                    }
                }
                stream_data_out.write(result_tmp);
            }
        }
    }
}

void adder_out(float *data_out,
               hls::stream<float> &stream_data_out,
               int p) {
    for (int i = 0; i < (image_height-filter_h+1)*(image_width-filter_w+1); i++) {
        data_out[p*(image_height-filter_h+1)*(image_width-filter_w+1)+i] += stream_data_out.read();
    }
}

void compute_one_plan(float *data_in, 
                      float *filter,
                      float *data_out, 
                      int s,
                      int p) {
    #pragma HLS inline off
    #pragma HLS dataflow       
    hls::stream<float> stream_data_out;                      
    compute_one_plan_sub(data_in, filter, stream_data_out, s, p);
    adder_out(data_out, stream_data_out, p);                      
}

void compute_cube(float *data_in, float *filter, float * data_out) {
    for (int p = 0; p < image_depth-filter_d+1; p++) {
        for (int s = 0; s < filter_d; s++) {
            compute_one_plan(data_in, filter, data_out, s, p);
        }
    }
}

#pragma ACCEL kernel
void conv_3d_kernel(float data_in[DATA_IN_LENGTH], float filter[FILTER_LENGTH], float data_out[DATA_OUT_LENGTH]){
    compute_cube(data_in, filter, data_out);
}
