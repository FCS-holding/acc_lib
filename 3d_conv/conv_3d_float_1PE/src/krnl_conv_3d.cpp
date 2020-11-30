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

    int i, j, w, h;
    float tmp1[filter_h];
    float tmp2[filter_w];
    float input_tmp[filter_h];
    float shift[filter_w][filter_h];

    #pragma ACCEL parallel complete
    for (w = 0; w < filter_w; w++) {
        for (h = 0; h < filter_h; h++) {
            shift[w][h] = ((float )0);
        }
    }

    for (int t = 0; t < image_height-filter_h+1; t++) {
        #pragma ACCEL pipeline flatten       
        for (i = 0; i < image_width; i++) {
            for (j = 0; j < filter_h; j++) {
                input_tmp[j] = input[j][i];
            }
            for (w = filter_w-1; w > 0; w--) {
                for (h = 0; h < filter_h; h++) {
                    shift[w][h] = shift[w-1][h];
                }
            }

            for (h = 0; h < filter_h-1; h++) {
                input[h][i] = input[h+1][i];
            }

            input[filter_h-1][i] = stream_data_in.read();
            for (h = 0; h < filter_h; h++) {
                shift[0][h] = input_tmp[h];
            }

            if (i >= filter_h - 1) {
                float result_tmp;
                for (w = 0; w < filter_w; w++) {
                    for (h = 0; h < filter_h; h++) {
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
    for (int i = 0; i < image_height-filter_h+1; ++i) {
        float buf[image_width];
        STREAM2:
        for (int j = 0; j < image_width; ++j) {
            buf[j] = data_in[(s_p)*image_width*image_height+image_width*filter_h+i*image_width+j];
            stream_data_in.write(buf[j]);
        }
    }
}

void adder_out(float plane_out[(image_height-filter_h+0)*(image_width-filter_w+1)],
               hls::stream<float> &stream_data_out) {

    for (int i = 0; i < (image_height-filter_h+1)*(image_width-filter_w+1); i++) {
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
    merlinL21:
    for (int j = 0; j < filter_h; j++) {
        merlinL20:
        for (int i = 0; i < image_width; i++) {
            data_in_tmp[j][i] = data_in[s_p * image_width * image_height + j * image_width + i];
        }
    }
}

void memory_burst_out(float plane_out[(image_height-filter_h+1)*(image_width-filter_w+1)],float *merlin_output, int p) {
    merlin16:
    for (int i = 0; i < (image_height-filter_h+1)*(image_width-filter_w+1); i++) {
        merlin_output[p*(image_height-filter_h+1)*(image_width-filter_w+1)+i] = plane_out[i];
    }
}

void init_plane_out(float plane_out[(image_height-filter_h+1)*(image_width-filter_w+1)]){
    for (int i = 0; i < (image_height-filter_h+1)*(image_width-filter_w+1); i++) {
        plane_out[i] = 0;
    }
}

void compute_cube(float *data_in, float *filter, float * data_out) {

    float filter_tmp[filter_w][filter_d][filter_h];
    #pragma HLS RESOURCE variable=filter_tmp core=RAM_2P_BRAM
    memory_burst_filter(filter_tmp, filter);
    
    for (int p = 0; p < image_depth-filter_d+1; p++) {
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



#pragma ACCEL kernel
void conv_3d_kernel(float data_in[DATA_IN_LENGTH], float filter[FILTER_LENGTH], float data_out[DATA_OUT_LENGTH]){

    compute_cube(data_in, filter, data_out);

}
