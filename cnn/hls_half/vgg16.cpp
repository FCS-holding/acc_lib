#include "falconML.hpp"
#include "vgg16.hpp"

#if EXTN
extern "C" {
#endif

void vgg16(ap_uint<BitW> *m_fm)
{ 
#pragma HLS interface m_axi offset=slave depth=2000000 port=m_fm
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=m_fm bundle=control

static data_t in[HWFIn][HWFinR][HWFinC];
static data_t Cout[HWFOut][HWFR][HWFC];
//static data_t Pout[HWFOut][HWFR/2][HWFC/2];
static data_t weight[HWFOut][HWFIn][HWKsize][HWKsize];
static data_t bias[512];
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=Cout complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2

#pragma HLS RESOURCE variable=weight core=RAM_2P_LUTRAM
#pragma HLS RESOURCE variable=bias core=RAM_2P_LUTRAM

static data_t in_1[HWFIn][HWFinR][HWFinC];
static data_t Cout_1[HWFOut][HWFR][HWFC];
static data_t weight_1[HWFOut][HWFIn][HWKsize][HWKsize];
#pragma HLS ARRAY_PARTITION variable=in_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=Cout_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_1 complete dim=2

#pragma HLS RESOURCE variable=weight_1 core=RAM_2P_LUTRAM

int ly[20][15]; 
#pragma HLS ARRAY_PARTITION variable=ly complete dim=2
for(int i=0; i<20; i++){
#pragma HLS pipeline
   bw_t tmp = m_fm[i]; 
   for(int j=0; j<15; j++){
       int data = tmp.range(2*DSIZE*(j+1)-1, 2*DSIZE*j);
       ly[i][j] = (int)data;
   }
}

int num_layer = ly[0][14];

for(int i=1; i< num_layer+1; i++) {
printf("layer %d execution, \n", i);
ConvKernel2<data_t, bw_t, HWFIn, HWFOut, HWFR, HWFC, HWFinR, HWFinC, HWKsize, HWin, HWout>
(in, Cout, weight, in_1, Cout_1, weight_1, bias, ly[i][0], ly[i][1], ly[i][4], ly[i][5], ly[i][2], ly[i][3], ly[i][6], ly[i][7], ly[i][8], ly[i][9], m_fm, ly[i][10], ly[i][11], ly[i][12], ly[i][13], ly[i][14]);

//ConvKernel2<data_t, bw_t, HWFIn, HWFOut, HWFR, HWFC, HWFinR, HWFinC, HWKsize, HWin, HWout>
//(in, Cout, weight, in_1, Cout_1, weight_1, bias, ly[i].fin, ly[i].fout, ly[i].frow, ly[i].fcol, ly[i].finrow, ly[i].fincol, ly[i].Ksize, ly[i].Kstride, ly[i].pad, ly[i].mask, m_fm, ly[i].addr_in, ly[i].addr_wght, ly[i].addr_out, ly[i].pool, ly[i].relu);

}

/*
//+++++++++++++++++ layer 1_1 +++++++++++++++++++++++++++++++
//------------- convolution ---------------
ConvKernel2<data_t, bw_t, HWFIn, HWFOut, HWFR, HWFC, HWFinR, HWFinC, HWKsize, HWin, HWout>
(in, Cout, weight, in_1, Cout_1, weight_1, bias, conv1_1.fin, conv1_1.fout, conv1_1.frow, conv1_1.fcol, conv1_1.finrow, conv1_1.fincol, conv1_1.Ksize, conv1_1.Kstride, conv1_1.pad, conv1_1.mask, m_fm, conv1_1.addr_in, conv1_1.addr_wght, conv1_1.addr_out, conv1_1.pool, conv1_1.relu);

//+++++++++++++++++ layer 1_2 +++++++++++++++++++++++++++++++
//------------- convolution ---------------
ConvKernel2<data_t, bw_t, HWFIn, HWFOut, HWFR, HWFC, HWFinR, HWFinC, HWKsize, HWin, HWout>
(in, Cout, weight, in_1, Cout_1, weight_1, bias, conv1_2.fin, conv1_2.fout, conv1_2.frow, conv1_2.fcol, conv1_2.finrow, conv1_2.fincol, conv1_2.Ksize, conv1_2.Kstride, conv1_2.pad, conv1_2.mask, m_fm, conv1_2.addr_in, conv1_2.addr_wght, conv1_2.addr_out, conv1_2.pool, conv1_2.relu);

//+++++++++++++++++ layer 2_1 +++++++++++++++++++++++++++++++
//------------- convolution ---------------
ConvKernel2<data_t, bw_t, HWFIn, HWFOut, HWFR, HWFC, HWFinR, HWFinC, HWKsize, HWin, HWout>
(in, Cout, weight, in_1, Cout_1, weight_1, bias, conv2_1.fin, conv2_1.fout, conv2_1.frow, conv2_1.fcol, conv2_1.finrow, conv2_1.fincol, conv2_1.Ksize, conv2_1.Kstride, conv2_1.pad, conv2_1.mask, m_fm, conv2_1.addr_in, conv2_1.addr_wght, conv2_1.addr_out, conv2_1.pool, conv2_1.relu);

//+++++++++++++++++ layer 2_2 +++++++++++++++++++++++++++++++
//------------- convolution ---------------
ConvKernel2<data_t, bw_t, HWFIn, HWFOut, HWFR, HWFC, HWFinR, HWFinC, HWKsize, HWin, HWout>
(in, Cout, weight, in_1, Cout_1, weight_1, bias, conv2_2.fin, conv2_2.fout, conv2_2.frow, conv2_2.fcol, conv2_2.finrow, conv2_2.fincol, conv2_2.Ksize, conv2_2.Kstride, conv2_2.pad, conv2_2.mask, m_fm, conv2_2.addr_in, conv2_2.addr_wght, conv2_2.addr_out, conv2_2.pool, conv2_2.relu);

//+++++++++++++++++ layer 3_1 +++++++++++++++++++++++++++++++
//------------- convolution ---------------
ConvKernel2<data_t, bw_t, HWFIn, HWFOut, HWFR, HWFC, HWFinR, HWFinC, HWKsize, HWin, HWout>
(in, Cout, weight, in_1, Cout_1, weight_1, bias, conv3_1.fin, conv3_1.fout, conv3_1.frow, conv3_1.fcol, conv3_1.finrow, conv3_1.fincol, conv3_1.Ksize, conv3_1.Kstride, conv3_1.pad, conv3_1.mask, m_fm, conv3_1.addr_in, conv3_1.addr_wght, conv3_1.addr_out, conv3_1.pool, conv3_1.relu);

//+++++++++++++++++ layer 3_2 +++++++++++++++++++++++++++++++
//------------- convolution ---------------
ConvKernel2<data_t, bw_t, HWFIn, HWFOut, HWFR, HWFC, HWFinR, HWFinC, HWKsize, HWin, HWout>
(in, Cout, weight, in_1, Cout_1, weight_1, bias, conv3_2.fin, conv3_2.fout, conv3_2.frow, conv3_2.fcol, conv3_2.finrow, conv3_2.fincol, conv3_2.Ksize, conv3_2.Kstride, conv3_2.pad, conv3_2.mask, m_fm, conv3_2.addr_in, conv3_2.addr_wght, conv3_2.addr_out, conv3_2.pool, conv3_2.relu);

//+++++++++++++++++ layer 3_3 +++++++++++++++++++++++++++++++
//------------- convolution ---------------
ConvKernel2<data_t, bw_t, HWFIn, HWFOut, HWFR, HWFC, HWFinR, HWFinC, HWKsize, HWin, HWout>
(in, Cout, weight, in_1, Cout_1, weight_1, bias, conv3_3.fin, conv3_3.fout, conv3_3.frow, conv3_3.fcol, conv3_3.finrow, conv3_3.fincol, conv3_3.Ksize, conv3_3.Kstride, conv3_3.pad, conv3_3.mask, m_fm, conv3_3.addr_in, conv3_3.addr_wght, conv3_3.addr_out, conv3_3.pool, conv3_3.relu);

//+++++++++++++++++ layer 4_1 +++++++++++++++++++++++++++++++
//------------- convolution ---------------
ConvKernel2<data_t, bw_t, HWFIn, HWFOut, HWFR, HWFC, HWFinR, HWFinC, HWKsize, HWin, HWout>
(in, Cout, weight, in_1, Cout_1, weight_1, bias, conv4_1.fin, conv4_1.fout, conv4_1.frow, conv4_1.fcol, conv4_1.finrow, conv4_1.fincol, conv4_1.Ksize, conv4_1.Kstride, conv4_1.pad, conv4_1.mask, m_fm, conv4_1.addr_in, conv4_1.addr_wght, conv4_1.addr_out, conv4_1.pool, conv4_1.relu);

//+++++++++++++++++ layer 4_2 +++++++++++++++++++++++++++++++
//------------- convolution ---------------
ConvKernel2<data_t, bw_t, HWFIn, HWFOut, HWFR, HWFC, HWFinR, HWFinC, HWKsize, HWin, HWout>
(in, Cout, weight, in_1, Cout_1, weight_1, bias, conv4_2.fin, conv4_2.fout, conv4_2.frow, conv4_2.fcol, conv4_2.finrow, conv4_2.fincol, conv4_2.Ksize, conv4_2.Kstride, conv4_2.pad, conv4_2.mask, m_fm, conv4_2.addr_in, conv4_2.addr_wght, conv4_2.addr_out, conv4_2.pool, conv4_2.relu);

//+++++++++++++++++ layer 4_3 +++++++++++++++++++++++++++++++
//------------- convolution ---------------
ConvKernel2<data_t, bw_t, HWFIn, HWFOut, HWFR, HWFC, HWFinR, HWFinC, HWKsize, HWin, HWout>
(in, Cout, weight, in_1, Cout_1, weight_1, bias, conv4_3.fin, conv4_3.fout, conv4_3.frow, conv4_3.fcol, conv4_3.finrow, conv4_3.fincol, conv4_3.Ksize, conv4_3.Kstride, conv4_3.pad, conv4_3.mask, m_fm, conv4_3.addr_in, conv4_3.addr_wght, conv4_3.addr_out, conv4_3.pool, conv4_3.relu);

//+++++++++++++++++ layer 5_1 +++++++++++++++++++++++++++++++
//------------- convolution ---------------
ConvKernel2<data_t, bw_t, HWFIn, HWFOut, HWFR, HWFC, HWFinR, HWFinC, HWKsize, HWin, HWout>
(in, Cout, weight, in_1, Cout_1, weight_1, bias, conv5_1.fin, conv5_1.fout, conv5_1.frow, conv5_1.fcol, conv5_1.finrow, conv5_1.fincol, conv5_1.Ksize, conv5_1.Kstride, conv5_1.pad, conv5_1.mask, m_fm, conv5_1.addr_in, conv5_1.addr_wght, conv5_1.addr_out, conv5_1.pool, conv5_1.relu);

//+++++++++++++++++ layer 5_2 +++++++++++++++++++++++++++++++
//------------- convolution ---------------
ConvKernel2<data_t, bw_t, HWFIn, HWFOut, HWFR, HWFC, HWFinR, HWFinC, HWKsize, HWin, HWout>
(in, Cout, weight, in_1, Cout_1, weight_1, bias, conv5_2.fin, conv5_2.fout, conv5_2.frow, conv5_2.fcol, conv5_2.finrow, conv5_2.fincol, conv5_2.Ksize, conv5_2.Kstride, conv5_2.pad, conv5_2.mask, m_fm, conv5_2.addr_in, conv5_2.addr_wght, conv5_2.addr_out, conv5_2.pool, conv5_2.relu);

//+++++++++++++++++ layer 5_3 +++++++++++++++++++++++++++++++
//------------- convolution ---------------
ConvKernel2<data_t, bw_t, HWFIn, HWFOut, HWFR, HWFC, HWFinR, HWFinC, HWKsize, HWin, HWout>
(in, Cout, weight, in_1, Cout_1, weight_1, bias, conv5_3.fin, conv5_3.fout, conv5_3.frow, conv5_3.fcol, conv5_3.finrow, conv5_3.fincol, conv5_3.Ksize, conv5_3.Kstride, conv5_3.pad, conv5_3.mask, m_fm, conv5_3.addr_in, conv5_3.addr_wght, conv5_3.addr_out, conv5_3.pool, conv5_3.relu);
*/

} // brace of main

#if EXTN
}
#endif


