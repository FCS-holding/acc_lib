#include <stdio.h>
#include <hls_stream.h>
#include "conv_3d.h"
void fadd_parallel(float *in, float *out) {
    float tmp=0;                             
    for(int i = 0; i < FILTER_SIZE; i++){               
        tmp += in[i];                          
    }                                          
    *out = tmp;                                
}
void compute_one_plan_sub(float input[FILTER_SIZE][IMAGE_SIZE], 
                          hls::stream<float> &stream_data_in, 
                          float filter[FILTER_SIZE][FILTER_SIZE],
                          hls::stream<float> &stream_data_out) {
    int i, w, h;
    float tmp1[FILTER_SIZE];
    float tmp2[FILTER_SIZE];
    float window[FILTER_SIZE][FILTER_SIZE];
    #pragma ACCEL parallel complete
    for (w = 0; w < FILTER_SIZE; w++) {
        for (h = 0; h < FILTER_SIZE; h++) {
            window[w][h] = ((float )0);
        }
    }
    //for (int t = 0; t < OUT_IMAGE_SIZE; t++) {
    for (int t = 0; t < STEP; t++) {
        #pragma ACCEL pipeline flatten       
        for (i = 0; i < IMAGE_SIZE; i++) {
            float tmp = stream_data_in.read();
            // shift top the column i of input, so the bottom data will be empty, waiting for next streaming
            for (h = 0; h < FILTER_SIZE-1; h++) {
                input[h][i] = input[h+1][i];
            }
            // shift in one data to last element of buf
            input[FILTER_SIZE-1][i] = tmp;
            // left shift the whole window
            for (w = FILTER_SIZE-1; w > 0; w--) {
                for (h = 0; h < FILTER_SIZE; h++) {
                    window[w][h] = window[w-1][h];
                }
            }
            // shift buf to the first column into windows
            for (h = 0; h < FILTER_SIZE; h++) {
                window[0][h] = input[h][i];
            }
            if (i >= FILTER_SIZE - 1) {
                // do computition with windows and filters
                float result_tmp;
                for (w = 0; w < FILTER_SIZE; w++) {
                    for (h = 0; h < FILTER_SIZE; h++) {
                        #pragma HLS RESOURCE variable=result_tmp core=FMul_fulldsp
                        result_tmp = window[FILTER_SIZE - w - 1][h] * filter[w][h];
                        //if (i == FILTER_SIZE - 1 && t == 0) {
                        //    printf("w = %d, h = %d, data = %f, filter = %f\n", w, h, window[FILTER_SIZE - w - 1][h], filter[w][h]);
                        //}
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
                   int s) {
    for (int i = 0; i < STEP; ++i) {
        float buf[IMAGE_SIZE];
        for (int j = 0; j < IMAGE_SIZE; ++j) {
            buf[j] = data_in[s*IN_DEPTH*IMAGE_SIZE+IMAGE_SIZE*(FILTER_SIZE+i-1)+j];
            stream_data_in.write(buf[j]);
        }
    }
}

void adder_out(float plane_out[STEP*OUT_IMAGE_SIZE],
               hls::stream<float> &stream_data_out) {
    for (int i = 0; i < STEP*OUT_IMAGE_SIZE; i++) {
        plane_out[i] += stream_data_out.read();
    }
}

void compute_one_plan(float data_in_tmp[FILTER_SIZE][IMAGE_SIZE],
                      float *data_in, 
                      float filter[FILTER_SIZE][FILTER_SIZE],
                      float plane_out[STEP*OUT_IMAGE_SIZE], 
                      int s) {
    #pragma HLS inline off
    #pragma HLS dataflow       
    hls::stream<float> stream_data_in;
    hls::stream<float> stream_data_out;                      
    shift_in_data(data_in, stream_data_in, s);                      
    compute_one_plan_sub(data_in_tmp, stream_data_in, filter, stream_data_out);
    adder_out(plane_out, stream_data_out);                      
}

void memory_burst_filter(float filter_tmp[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE],float *filter) {
    for (int i = 0; i < FILTER_SIZE; i++) {
        for (int j = 0; j < FILTER_SIZE; j++) {
            #pragma ACCEL pipeline flatten
            for (int k = 0; k < FILTER_SIZE; k++) {
                filter_tmp[i][j][k] = filter[i*FILTER_SIZE*FILTER_SIZE+k*FILTER_SIZE+j];
            }
        }
    }
}

void memory_burst_in(float data_in_tmp[FILTER_SIZE][IMAGE_SIZE],float *data_in, int s) {
    for (int j = 0; j < FILTER_SIZE-1; j++) {
        for (int i = 0; i < IMAGE_SIZE; i++) {
            data_in_tmp[j+1][i] = data_in[s*IN_DEPTH*IMAGE_SIZE + j*IMAGE_SIZE + i];
        }
    }
}

void memory_burst_out(float plane_out[STEP*OUT_IMAGE_SIZE],float *merlin_output) {
    for (int i = 0; i < STEP*OUT_IMAGE_SIZE; i++) {
        merlin_output[i] = plane_out[i];
    }
}

void init_plane_out(float plane_out[STEP*OUT_IMAGE_SIZE]){
    for (int i = 0; i < STEP*OUT_IMAGE_SIZE; i++) {
        plane_out[i] = 0;
    }
}

#pragma ACCEL kernel
void conv_3d_kernel(float data_in[IN_SIZE_ONE_CALL], float filter[FILTER_IN_LENGTH], float data_out[OUT_SIZE_ONE_CALL]){
    float filter_tmp[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE];
    float plane_out[STEP*OUT_IMAGE_SIZE];
    #pragma HLS RESOURCE variable=filter_tmp core=RAM_2P_BRAM
    memory_burst_filter(filter_tmp, filter);
    init_plane_out(plane_out);
    for (int s = 0; s < FILTER_SIZE; s++) {
//        printf("s = %d\n", s);
        float data_in_tmp[FILTER_SIZE][IMAGE_SIZE];
        #pragma HLS RESOURCE variable=plane_out core=RAM_2P_URAM
        memory_burst_in(data_in_tmp, data_in, s);
        compute_one_plan(data_in_tmp, data_in, filter_tmp[s], plane_out, s);
    }
    memory_burst_out(plane_out, data_out);
//    printf("index = %d, data_out_kernel[%d] = %f\n", 0, 0, data_out[0]);
}
