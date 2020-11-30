#include "cnn_cfg.hpp"
#include "falconML.hpp"

#if EXTN
extern "C" {
#endif

void vgg16(ap_uint<BitW> *m_fm, ap_uint<512> *lyinf)
{ 
#pragma HLS interface m_axi offset=slave depth=2000000 port=m_fm bundle=m_1
#pragma HLS interface m_axi offset=slave depth=2000 port=lyinf bundle=m_2
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=m_fm bundle=control
#pragma HLS INTERFACE s_axilite port=lyinf bundle=control

static data_t in[HWFIn][HWFinR][HWFinC];
static data_t Cout[HWFOut][HWFR][HWFC];
static data_t Cout_1[HWFOut][HWFR][HWFC];
static wght_t weight[HWFOut][HWFIn][HWKsize][HWKsize];
static wght_t bias[512];
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=Cout complete dim=1
#pragma HLS ARRAY_PARTITION variable=Cout_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2

static data_t in_1[HWFIn][HWFinR][HWFinC];
static wght_t weight_1[HWFOut][HWFIn][HWKsize][HWKsize];
#pragma HLS ARRAY_PARTITION variable=in_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_1 complete dim=2

int ly[20][16]; 
#pragma HLS ARRAY_PARTITION variable=ly complete dim=2

#pragma HLS RESOURCE variable=in core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=in_1 core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=Cout core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=Cout_1 core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=weight_1 core=RAM_2P_LUTRAM
#pragma HLS RESOURCE variable=weight core=RAM_2P_LUTRAM

for(int i=0; i<20; i++){
#pragma HLS pipeline
   bw_t tmp = lyinf[i]; 
   for(int j=0; j<16; j++){
       int data = tmp.range(32*(j+1)-1, 32*j);
       ly[i][j] = (int)data;
   }
}
printf("finish load layer definition\n");

int num_layer = ly[0][14];

for(int i=1; i< num_layer+1; i++) {

ConvKernel2<data_t, bw_t, HWFIn, HWFOut, HWFR, HWFC, HWFinR, HWFinC, HWKsize, HWin, HWout>
(in, Cout, weight, in_1, Cout_1, weight_1, bias, ly[i][0], ly[i][1], ly[i][4], ly[i][5], ly[i][2], ly[i][3], ly[i][6], ly[i][7], ly[i][8], ly[i][9], m_fm, ly[i][10], ly[i][11], ly[i][12], ly[i][13], ly[i][14]);


}


} // brace of main


#if EXTN
}
#endif

