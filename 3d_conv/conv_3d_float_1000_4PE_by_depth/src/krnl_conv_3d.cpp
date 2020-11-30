#include <stdio.h>
#include <hls_stream.h>
#include "conv_3d.h"
void fadd_parallel(float *in, float *out) {
    float in_tmp_1 = in[0] + in[1];
    float in_tmp_2 = in[2] + in[3];
    float in_tmp_3 = in[4] + in[5];
    float in_tmp_4 = in[6] + in[7];
    float in_tmp_5 = in[8] + in[9];
    float in_tmp_6 = in[10] + in[11];
    float in_tmp_7 = in[12] + in[13];
    float in_tmp_8 = in[14] + in[15];
    float in_tmp_9 = in[16] + in[17];
    float in_tmp_10 = in[18] + in[19];
    float in_tmp_11 = in[20] + in[21];
    float in_tmp_12 = in[22] + in[23];
    float in_tmp_1_1 = in_tmp_1 + in_tmp_2;
    float in_tmp_1_2 = in_tmp_3 + in_tmp_4;
    float in_tmp_1_3 = in_tmp_5 + in_tmp_6;
    float in_tmp_1_4 = in_tmp_7 + in_tmp_8;
    float in_tmp_1_5 = in_tmp_9 + in_tmp_10;
    float in_tmp_1_6 = in_tmp_11 + in_tmp_12;
    float in_tmp_1_1_1 = in_tmp_1_1 + in_tmp_1_2;
    float in_tmp_1_1_2 = in_tmp_1_3 + in_tmp_1_4;
    float in_tmp_1_1_3 = in_tmp_1_5 + in_tmp_1_6;
    *out = in_tmp_1_1_1 + in_tmp_1_1_2 + in_tmp_1_1_3;
}
void compute_one_plan_sub(float input[filter_h][image_width], 
                          hls::stream<float> &stream_data_in, 
                          float filter[filter_w][filter_h],
                          hls::stream<float> &stream_data_out) {
    #pragma HLS inline off
    float shift[filter_w][filter_h];
    #pragma HLS array_partition variable=shift complete dim=1
    #pragma HLS array_partition variable=shift complete dim=2

    #pragma ACCEL parallel complete
    for (int w = 0; w < filter_w; w++) {
        for (int h = 0; h < filter_h; h++) {
            shift[w][h] = ((float )0);
        }
    }

    for (int t = 0; t < image_height-filter_h+1; t++) {
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
                #pragma HLS array_partition variable=tmp2 complete dim=1
                for (int w = 0; w < filter_w; w++) {
                    float tmp_result;
                    #pragma HLS resource variable=tmp_result core=FMul_meddsp latency=4
                    float tmp1[filter_h];
                    for (int h = 0; h < filter_h; h++) {
                        #pragma HLS resource variable=tmp_result core=FMul_meddsp latency=4
                        tmp_result = shift[filter_w - w - 1][h] * filter[w][h];
                        tmp1[h] = tmp_result;
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
    #pragma HLS inline off
    for (int i = 0; i < image_height-filter_h+1; ++i) {
        float buf[image_width];
        for (int j = 0; j < image_width; ++j) {
            buf[j] = data_in[(s_p)*image_width*image_height+image_width*filter_h+i*image_width+j];
            stream_data_in.write(buf[j]);
        }
    }
}

void adder_out(float plane_out[(image_height-filter_h+0)*(image_width-filter_w+1)],
               hls::stream<float> &stream_data_out) {
    #pragma HLS inline off
    for (int i = 0; i < (image_height-filter_h+1)*(image_width-filter_w+1); i++) {
        #pragma HLS pipeline
        plane_out[i] += stream_data_out.read();
    }
}

void compute_one_plan(float data_in_tmp[filter_h][image_width],
                      float *data_in, 
                      float filter[filter_w][filter_h],
                      float plane_out[(image_height-filter_h+1)*(image_width-filter_w+1)], 
                      int s_p) {
    #pragma HLS inline off
    #pragma HLS dataflow       

    hls::stream<float> stream_data_in;
    #pragma HLS STREAM variable=stream_data_in depth=image_width
    hls::stream<float> stream_data_out;                      
    #pragma HLS STREAM variable=stream_data_out depth=image_width

    shift_in_data(data_in, stream_data_in, s_p);                      
    compute_one_plan_sub(data_in_tmp, stream_data_in, filter, stream_data_out);
    adder_out(plane_out, stream_data_out);                      
}

void memory_burst_filter(float filter_tmp[filter_d][filter_w][filter_h],float *filter) {
    #pragma HLS inline off
    for (int i = 0; i < filter_d; i++) {
        for (int j = 0; j < filter_w; j++) {
            for (int k = 0; k < filter_h; k++) {
                #pragma HLS pipeline
                filter_tmp[i][j][k] = filter[i*filter_w*filter_h+k*filter_h+j];
            }
        }
    }
}

void memory_burst_in(float data_in_tmp[filter_h][image_width],float *data_in, int s_p) {
    #pragma HLS inline off
    for (int j = 0; j < filter_h; j++) {
        for (int i = 0; i < image_width; i++) {
            #pragma HLS pipeline
            data_in_tmp[j][i] = data_in[s_p * image_width * image_height + j * image_width + i];
        }
    }
}

void memory_burst_out(float plane_out[(image_height-filter_h+1)*(image_width-filter_w+1)],float *merlin_output, int p) {
    #pragma HLS inline off
    for (int i = 0; i < (image_height-filter_h+1)*(image_width-filter_w+1); i++) {
        merlin_output[p*(image_height-filter_h+1)*(image_width-filter_w+1)+i] = plane_out[i];
    }
}

void init_plane_out(float plane_out[(image_height-filter_h+1)*(image_width-filter_w+1)]){
    #pragma HLS inline off 
    for (int i = 0; i < (image_height-filter_h+1)*(image_width-filter_w+1); i++) {
        plane_out[i] = 0;
    }
}

#pragma ACCEL kernel
void conv_3d_kernel0(float data_in[IN_SIZE_0],
                     float filter[FILTER_LENGTH],
                     float data_out[OUT_SIZE_0]){
    #pragma ACCEL interface variable=data_in    bank=0
    #pragma ACCEL interface variable=filter     bank=0
    #pragma ACCEL interface variable=data_out   bank=0
    float filter_tmp[filter_w][filter_d][filter_h];
    #pragma HLS RESOURCE variable=filter_tmp core=RAM_2P_LUTRAM
    memory_burst_filter(filter_tmp, filter);
    for (int p = 0; p < OUT_DEPTH_0; p++) {
        float plane_out[(image_height-filter_h+1)*(image_width-filter_w+1)];
        init_plane_out(plane_out);
        for (int s = 0; s < filter_d; s++) {
            int s_p = s + p;
            float data_in_tmp[filter_h][image_width];
            #pragma HLS RESOURCE variable=plane_out core=RAM_2P_URAM latency=4
            memory_burst_in(data_in_tmp, data_in, s_p);
            compute_one_plan(data_in_tmp, data_in, filter_tmp[s], plane_out, s_p);
        }
        memory_burst_out(plane_out, data_out, p);
    }
}

#if PE>=2
#pragma ACCEL kernel
void conv_3d_kernel1(float data_in[IN_SIZE_1],
                     float filter[FILTER_LENGTH],
                     float data_out[OUT_SIZE_1]){
    #pragma ACCEL interface variable=data_in    bank=1
    #pragma ACCEL interface variable=filter     bank=1
    #pragma ACCEL interface variable=data_out   bank=1
    float filter_tmp[filter_w][filter_d][filter_h];
    #pragma HLS RESOURCE variable=filter_tmp core=RAM_2P_LUTRAM
    memory_burst_filter(filter_tmp, filter);
    for (int p = 0; p < OUT_DEPTH_1; p++) {
        float plane_out[(image_height-filter_h+1)*(image_width-filter_w+1)];
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
#endif

#if PE>=4
#pragma ACCEL kernel
void conv_3d_kernel2(float data_in[IN_SIZE_2],
                     float filter[FILTER_LENGTH],
                     float data_out[OUT_SIZE_2]){
    #pragma ACCEL interface variable=data_in    bank=2
    #pragma ACCEL interface variable=filter     bank=2
    #pragma ACCEL interface variable=data_out   bank=2
    float filter_tmp[filter_w][filter_d][filter_h];
    #pragma HLS RESOURCE variable=filter_tmp core=RAM_2P_LUTRAM
    memory_burst_filter(filter_tmp, filter);
    for (int p = 0; p < OUT_DEPTH_2; p++) {
        float plane_out[(image_height-filter_h+1)*(image_width-filter_w+1)];
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
#endif

#if PE>=4
#pragma ACCEL kernel
void conv_3d_kernel3(float data_in[IN_SIZE_3],
                     float filter[FILTER_LENGTH],
                     float data_out[OUT_SIZE_3]){
    #pragma ACCEL interface variable=data_in    bank=3
    #pragma ACCEL interface variable=filter     bank=3
    #pragma ACCEL interface variable=data_out   bank=3
    float filter_tmp[filter_w][filter_d][filter_h];
    #pragma HLS RESOURCE variable=filter_tmp core=RAM_2P_LUTRAM
    memory_burst_filter(filter_tmp, filter);
    for (int p = 0; p < OUT_DEPTH_3; p++) {
        float plane_out[(image_height-filter_h+1)*(image_width-filter_w+1)];
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
