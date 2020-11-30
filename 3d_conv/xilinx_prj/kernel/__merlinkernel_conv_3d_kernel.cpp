#include "conv_3d.h"
#include <hls_stream.h>
static void memory_burst_filter_1(float filter_tmp[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE],float *filter) {
  merlinL1:
  for (int i = 0; i < FILTER_SIZE; i++) {
    int j;
    int k;
    long _j_k;
    j = 0;
    k = 0;
    merlinL0:
    for (_j_k = 0; _j_k <= FILTER_SIZE * FILTER_SIZE - 1; ++_j_k) {
      #pragma HLS pipeline
      filter_tmp[i][j][k] = filter[i * FILTER_SIZE * FILTER_SIZE + k * FILTER_SIZE + j];
      {
        ++k;
        if (k > FILTER_SIZE - 1) {
          ++j;
          if (j <= FILTER_SIZE - 1) 
            k = 0;
        }
      }
    }
  }
}
static void init_plane_out_1(float plane_out[STEP*OUT_IMAGE_SIZE]) {
  merlinL2:
  for (int i = 0; i < STEP*OUT_IMAGE_SIZE; i++) {
    #pragma HLS pipeline
    plane_out[i] = 0;
  }
}
static void memory_burst_in_1(float data_in_tmp[FILTER_SIZE][IMAGE_SIZE],float *data_in,int s) {
  merlinL4:
  for (int j = 0; j < FILTER_SIZE-1; j++) {
    merlinL3:
    for (int i = 0; i < IMAGE_SIZE; i++) {
      #pragma HLS pipeline
      data_in_tmp[j + 1][i] = data_in[s*IN_DEPTH*IMAGE_SIZE + j*IMAGE_SIZE + i];
    }
  }
}
static void shift_in_data_1(float *data_in,class hls::stream< float  > &stream_data_in,int s) {
  merlinL6:
  for (int i = 0; i < STEP; ++i) {
    float buf[IMAGE_SIZE];
    merlinL5:
    for (int j = 0; j < IMAGE_SIZE; ++j) {
      #pragma HLS pipeline
      buf[j] = data_in[s*IN_DEPTH*IMAGE_SIZE+IMAGE_SIZE*(FILTER_SIZE+i-1)+j];
      stream_data_in . write(buf[j]);
    }
  }
}

static void fadd_parallel_1(float *in,float *out) {
  #pragma HLS INLINE
  float tmp = 0;
  merlinL17:
  for (int i = 0; i < FILTER_SIZE; i++) {
    #pragma HLS unroll
    tmp += in[i];
  }
  *out = tmp;
}

static void compute_one_plan_sub_1(float input[FILTER_SIZE][IMAGE_SIZE],
                                   class hls::stream< float  > &stream_data_in,
                                   float filter[FILTER_SIZE][FILTER_SIZE],
                                   class hls::stream< float  > &stream_data_out) {
  int i;
  int w;
  int h;
  float tmp1[FILTER_SIZE];
  #pragma HLS array_partition variable=tmp1 complete dim=1
  float tmp2[FILTER_SIZE];
  #pragma HLS array_partition variable=tmp2 complete dim=1
  float window[FILTER_SIZE][FILTER_SIZE];
  #pragma HLS array_partition variable=window complete dim=2
  #pragma HLS array_partition variable=window complete dim=1
  merlinL16:
  for (w = 0; w < FILTER_SIZE; w++) {
    #pragma HLS unroll
    merlinL15:
    for (h = 0; h < FILTER_SIZE; h++) {
      #pragma HLS unroll
      window[w][h] = 0;
    }
  }
  merlinL14:
  for (int t = 0; t < STEP; t++) {
    merlinL13:
    for (i = 0; i < IMAGE_SIZE; i++) {
      #pragma HLS pipeline
      float tmp = stream_data_in . read();
      merlinL12:
      for (h = 0; h < FILTER_SIZE - 1; h++) {
        #pragma HLS unroll
        input[h][i] = input[h + 1][i];
      }
      input[FILTER_SIZE - 1][i] = tmp;
      merlinL11:
      for (w = 0; w <= FILTER_SIZE-2; w++) {
        #pragma HLS unroll
        int _in_w = FILTER_SIZE - 1 - w;
        merlinL10:
        for (h = 0; h < FILTER_SIZE; h++) {
          #pragma HLS unroll
          window[_in_w][h] = window[_in_w - 1][h];
        }
      }
      w = 1 + - 1;
      merlinL9:
      for (h = 0; h < FILTER_SIZE; h++) {
        #pragma HLS unroll
        window[0][h] = input[h][i];
      }
      if (i >= FILTER_SIZE - 1) {
        float result_tmp;
        merlinL8:
        for (w = 0; w < FILTER_SIZE; w++) {
          #pragma HLS unroll
          merlinL7:
          for (h = 0; h < FILTER_SIZE; h++) {
            #pragma HLS unroll
            #pragma HLS RESOURCE variable=result_tmp core=FMul_fulldsp
            result_tmp = window[FILTER_SIZE - w - 1][h] * filter[w][h];
            tmp1[h] = result_tmp;
          }
          fadd_parallel_1(tmp1,&tmp2[w]);
        }
        float tmp3;
        fadd_parallel_1(tmp2,&tmp3);
        stream_data_out . write(tmp3);
      }
    }
  }
}

static void adder_out_1(float plane_out[STEP*OUT_IMAGE_SIZE],class hls::stream< float  > &stream_data_out) {
  merlinL18:
  for (int i = 0; i < STEP*OUT_IMAGE_SIZE; i++) {
    #pragma HLS pipeline
    plane_out[i] += stream_data_out . read();
  }
}

static void compute_one_plan_1(float data_in_tmp[FILTER_SIZE][IMAGE_SIZE],
                               float *data_in,
                               float filter[FILTER_SIZE][FILTER_SIZE],
                               float plane_out[STEP*OUT_IMAGE_SIZE],int s) {
  #pragma HLS inline off
  #pragma HLS dataflow
  class hls::stream< float  > stream_data_in;
  class hls::stream< float  > stream_data_out;
  shift_in_data_1(data_in,stream_data_in,s);
  compute_one_plan_sub_1(data_in_tmp,stream_data_in,filter,stream_data_out);
  adder_out_1(plane_out,stream_data_out);
}
static void memory_burst_out_1(float plane_out[STEP*OUT_IMAGE_SIZE],float *merlin_output) {
  merlinL19:
  for (int i = 0; i < STEP*OUT_IMAGE_SIZE; i++) {
    #pragma HLS pipeline
    merlin_output[i] = plane_out[i];
  }
}
extern "C" { 
void conv_3d_kernel(float data_in[IN_SIZE_ONE_CALL],
                    float filter[FILTER_IN_LENGTH],
                    float data_out[OUT_SIZE_ONE_CALL]) {
  #pragma HLS INTERFACE m_axi port=data_in offset=slave depth=6432000
  #pragma HLS INTERFACE m_axi port=data_out offset=slave depth=239365
  #pragma HLS INTERFACE m_axi port=filter offset=slave depth=13824
  #pragma HLS INTERFACE s_axilite port=data_in bundle=control
  #pragma HLS INTERFACE s_axilite port=data_out bundle=control
  #pragma HLS INTERFACE s_axilite port=filter bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control
  float filter_tmp[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE];
  #pragma HLS array_partition variable=filter_tmp complete dim=3
  #pragma HLS array_partition variable=filter_tmp complete dim=2
  float plane_out[STEP*OUT_IMAGE_SIZE];
  #pragma HLS RESOURCE variable=filter_tmp core=RAM_2P_BRAM
  memory_burst_filter_1(filter_tmp,filter);
  init_plane_out_1(plane_out);
  merlinL20:
  for (int s = 0; s < FILTER_SIZE; s++) {
    float data_in_tmp[FILTER_SIZE][IMAGE_SIZE];
    #pragma HLS array_partition variable=data_in_tmp complete dim=1
    #pragma HLS RESOURCE variable=plane_out core=RAM_2P_URAM
    memory_burst_in_1(data_in_tmp,data_in,s);
    compute_one_plan_1(data_in_tmp,data_in,filter_tmp[s],plane_out,s);
  }
  memory_burst_out_1(plane_out,data_out);
}
}
