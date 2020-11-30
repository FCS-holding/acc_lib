#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
//#include <CL/opencl.h>
#include <sys/time.h>

#include <iostream> 
#include <fstream> 
#include <string> 

//#include "cnn_cfg.hpp"
#include "vgg16.hpp"

////////////////////////////////////////////////////////////////////////////////

using namespace std;

int addr_data    = (LayerDef+ WEIGHTSIZE);
int addr_out11   = (LayerDef+ WEIGHTSIZE+ INFM);
int addr_out12   = (LayerDef+ WEIGHTSIZE+ INFM+ FM11);
int addr_pool12  = (LayerDef+ WEIGHTSIZE+ INFM+ FM11 + FM12);
int addr_out21   = (LayerDef+ WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12);
int addr_out22   = (LayerDef+ WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21);
int addr_pool22  = (LayerDef+ WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22);
int addr_out31   = (LayerDef+ WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22);
int addr_out32   = (LayerDef+ WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31);
int addr_out33   = (LayerDef+ WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32);
int addr_pool33  = (LayerDef+ WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32+ FM33);
int addr_out41   = (LayerDef+ WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32+ FM33+ PL33);
int addr_out42   = (LayerDef+ WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32+ FM33+ PL33+ FM41);
int addr_out43   = (LayerDef+ WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32+ FM33+ PL33+ FM41+ FM42);
int addr_pool43  = (LayerDef+ WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32+ FM33+ PL33+ FM41+ FM42+ FM43);
int addr_out51   = (LayerDef+ WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32+ FM33+ PL33+ FM41+ FM42+ FM43+ PL43);
int addr_out52   = (LayerDef+ WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32+ FM33+ PL33+ FM41+ FM42+ FM43+ PL43+ FM51);
int addr_out53   = (LayerDef+ WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32+ FM33+ PL33+ FM41+ FM42+ FM43+ PL43+ FM51+ FM52);
int addr_pool53  = (LayerDef+ WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32+ FM33+ PL33+ FM41+ FM42+ FM43+ PL43+ FM51+ FM52+ FM53);

int addr_wght11 = LayerDef+ 0;
int addr_wght12 = LayerDef+ Layer11;
int addr_wght21 = LayerDef+ Layer12;
int addr_wght22 = LayerDef+ Layer21;
int addr_wght31 = LayerDef+ Layer22;
int addr_wght32 = LayerDef+ Layer31;
int addr_wght33 = LayerDef+ Layer32;
int addr_wght41 = LayerDef+ Layer33;
int addr_wght42 = LayerDef+ Layer41;
int addr_wght43 = LayerDef+ Layer42;
int addr_wght51 = LayerDef+ Layer43;
int addr_wght52 = LayerDef+ Layer51;
int addr_wght53 = LayerDef+ Layer52;


//-------------- in, out, ir, ic,  r,  c, k,  s, pad, mask ------- 
layer conv1_1 = {DATA, CONV11, CONV11_I, CONV11_I, CONV11_R, CONV11_R, KSIZE, STRIDE, PAD, DATA, addr_data, addr_wght11, addr_out11, 0, RELU};
layer conv1_2 = {CONV11, CONV12, CONV12_I, CONV12_I, CONV12_R, CONV12_R, KSIZE, STRIDE, PAD, HWin, addr_out11, addr_wght12, addr_pool12, POOL, RELU};
//layer pool1_2 = {CONV11, POOL12, CONV12_R, CONV12_R, POOL12_R, POOL12_R, 2, 2, 0, 0, addr_pool12, addr_wght21, addr_out21, 0, RELU};
layer conv2_1 = {POOL12, CONV21, CONV21_I, CONV21_I, CONV21_R, CONV21_R, KSIZE, STRIDE, PAD, HWin, addr_pool12, addr_wght21, addr_out21, 0, RELU};
layer conv2_2 = {CONV21, CONV22, CONV22_I, CONV22_I, CONV22_R, CONV22_R, KSIZE, STRIDE, PAD, HWin, addr_out21, addr_wght22, addr_pool22, POOL, RELU};
//layer pool2_2 = {CONV22, POOL22, CONV22_R, CONV22_R, POOL22_R, POOL22_R, 2, 2, 0, 0};
layer conv3_1 = {POOL22, CONV31, CONV31_I, CONV31_I, CONV31_R, CONV31_R, KSIZE, STRIDE, PAD, HWin, addr_pool22, addr_wght31, addr_out31, 0, RELU};
layer conv3_2 = {CONV31, CONV32, CONV32_I, CONV32_I, CONV32_R, CONV32_R, KSIZE, STRIDE, PAD, HWin, addr_out31, addr_wght32, addr_out32, 0, RELU};
layer conv3_3 = {CONV32, CONV33, CONV33_I, CONV33_I, CONV33_R, CONV33_R, KSIZE, STRIDE, PAD, HWin, addr_out32, addr_wght33, addr_pool33, POOL, RELU};
//layer pool3_3 = {CONV33, POOL33, CONV33_R, CONV33_R, POOL33_R, POOL33_R, 2, 2, 0, 0};
layer conv4_1 = {POOL33, CONV41, CONV41_I, CONV41_I, CONV41_R, CONV41_R, KSIZE, STRIDE, PAD, HWin, addr_pool33, addr_wght41, addr_out41, 0, RELU};
layer conv4_2 = {CONV41, CONV42, CONV42_I, CONV42_I, CONV42_R, CONV42_R, KSIZE, STRIDE, PAD, HWin, addr_out41, addr_wght42, addr_out42, 0, RELU};
layer conv4_3 = {CONV42, CONV43, CONV43_I, CONV43_I, CONV43_R, CONV43_R, KSIZE, STRIDE, PAD, HWin, addr_out42, addr_wght43, addr_pool43, POOL, RELU};
//layer pool4_3 = {CONV43, POOL43, CONV43_R, CONV43_R, POOL43_R, POOL43_R, 2, 2, 0, 0};
layer conv5_1 = {POOL43, CONV51, CONV51_I, CONV51_I, CONV51_R, CONV51_R, KSIZE, STRIDE, PAD, HWin, addr_pool43, addr_wght51, addr_out51, 0, RELU};
layer conv5_2 = {CONV51, CONV52, CONV52_I, CONV52_I, CONV52_R, CONV52_R, KSIZE, STRIDE, PAD, HWin, addr_out51, addr_wght52, addr_out52, 0, RELU};
layer conv5_3 = {CONV52, CONV53, CONV53_I, CONV53_I, CONV53_R, CONV53_R, KSIZE, STRIDE, PAD, HWin, addr_out52, addr_wght53, addr_pool53, POOL, RELU};
//layer pool5_3 = {CONV53, POOL53, CONV53_R, CONV53_R, POOL53_R, POOL53_R, 2, 2, 0, 0};

////////////////////////////////////////////////////////////////////////////////

data_t absub(data_t a, data_t b) {
    data_t tmp = (a>b)?(data_t)(a-b):(data_t)(b-a);
    return tmp;
}
data_t abadd(data_t a, data_t b) {
    data_t tmp1 = (a>(data_t)0)?(data_t)a:(data_t)((data_t)-1*a);
    data_t tmp2 = (b>(data_t)0)?(data_t)b:(data_t)((data_t)-1*b);
    data_t tmp3 = (data_t)(tmp1+tmp2)/(data_t)2;
    return tmp3;
}
float cmpratio(data_t a, data_t b) {
    float diff = (float)absub(a, b);
    float aver = (float)abadd(a, b);
    float ratio = diff/aver;
    return ratio;
}
template<typename data_t>
void memset_int(data_t *m, data_t val, int addr, int length) {
    for(int i=0; i<length; i++) {
        //data_t tmp = (data_t) rand();
        //tmp = (data_t)fmod(tmp, 3.0)/3.0;
        data_t tmp = (val+(data_t)i*0.001);
        m[i + addr] = ((int)rand()%2==0)?(tmp):(-1.0*tmp);//val + 0.05*(float)i; 
    }
}
template<typename data_t>
void memset_int_b(data_t *m, data_t val, int addr, int length) {
    for(int i=0; i<length; i++) {
        m[i + addr] = val ; 
    }
}
void reorder_weight(data_t weight[512][512][3][3], data_t bias[512], data_t *m, int addr, layer conv) {
	for(int i=0; i<conv.fout; i++) {
        for(int j=0; j<conv.fin; j++) {
            for(int k=0; k<3; k++) {
                for(int h=0; h<3; h++) {
                    weight[i][j][k][h] = m[ addr + i*conv.fin*9 + j*9 + k*3 + h ];
                }
            }
        }
        if(conv.fin<UNROLL) {
            for(int j=conv.fin; j<UNROLL; j++) {
                for(int k=0; k<3; k++) {
                    for(int h=0; h<3; h++) {
                        weight[i][j][k][h] = (data_t)0;
                    }
                }
            }
        }
	}
    for(int i=0; i<conv.fout; i++) {
        bias[i] = m[addr + conv.fin*conv.fout*9 + i]; 
    }
    int in_factor = (conv.fin>UNROLL)?conv.fin:UNROLL;
	//for(int ii=0; ii<conv.fout/UNROLL; ii++) {
	for(int ii=0; ii<conv.fout; ii+=UNROLL) {
	    //for(int jj=0; jj<in_factor/UNROLL; jj++) {
	    for(int jj=0; jj<in_factor; jj+=UNROLL) {
            for(int k=0; k<3; k++) {
                for(int h=0; h<3; h++) {
                    for(int i=0; i<HWFOut; i++) {
                        for(int j=0; j<HWFIn; j++) {
                            //m[addr + (ii*in_factor*9 + jj*9 + (k*3+h))*HWFOut*HWFIn + i*HWFIn+j] = weight[ii*UNROLL+i][jj*UNROLL+j][k][h];
                            m[addr + ii*in_factor*9 + jj*9*UNROLL + (k*3+h)*HWFOut*HWFIn + i*HWFIn+j] = weight[ii+i][jj+j][k][h];
                        }
                    }
                }
            }
        }
    }
    for(int i=0; i<conv.fout; i++) {
        m[addr + conv.fout*in_factor*9 + i] = bias[i]; 
    }
}
void prepare_weight(bw_t *DRAM_weight_reorder, data_t *DRAM_weight ) {
	static data_t weight[512][512][3][3];
	static data_t bias[512];
	reorder_weight(weight, bias, DRAM_weight, 0, conv1_1);
	reorder_weight(weight, bias, DRAM_weight, Layer11, conv1_2);
	reorder_weight(weight, bias, DRAM_weight, Layer12, conv2_1);
	reorder_weight(weight, bias, DRAM_weight, Layer21, conv2_2);
	reorder_weight(weight, bias, DRAM_weight, Layer22, conv3_1);
	reorder_weight(weight, bias, DRAM_weight, Layer31, conv3_2);
	reorder_weight(weight, bias, DRAM_weight, Layer32, conv3_3);
	reorder_weight(weight, bias, DRAM_weight, Layer33, conv4_1);
	reorder_weight(weight, bias, DRAM_weight, Layer41, conv4_2);
	reorder_weight(weight, bias, DRAM_weight, Layer42, conv4_3);
	reorder_weight(weight, bias, DRAM_weight, Layer43, conv5_1);
	reorder_weight(weight, bias, DRAM_weight, Layer51, conv5_2);
	reorder_weight(weight, bias, DRAM_weight, Layer52, conv5_3);
    bw_t tmp;
	for(int i=0; i<WEIGHTSIZE/UNROLL; i++) {
        for(int j=0; j<UNROLL; j++) {
            data_t data = DRAM_weight[i*UNROLL + j];
            int *idata = (int*)&data;
            tmp.range((j+1)*32-1, j*32) = *idata;
        }
        DRAM_weight_reorder[i] = tmp;
	}
}

void prepare_image(bw_t *m_fm, int addr_fm, data_t *m, int addr_m, int length) {
    int factor = UNROLL;
    bw_t data;
    for(int i=0; i<length/factor; i++) {
        for(int j=0; j<factor; j++) {
            data_t fdata = m[addr_m + i*factor + j];
            int* idata = (int*) &fdata;
            data.range((j+1)*32-1, j*32) = *idata;
        }
        m_fm[i + addr_fm] = data;
    }
	float adata[8][224][224];
    factor = sizeof(bw_t)/sizeof(float);
    bw_t tmp;
	for(int i=0; i<3; i++) {
	    for(int j=0; j<224; j++) {
	        for(int k=0; k<224; k+=factor){
                for(int kk=0; kk<factor; kk++) {
	                int tdata = m_fm[(i*224*224+j*224+k)/factor].range((kk+1)*32-1, kk*32);
                    float *fdata = (float*)&tdata;
	                adata[i][j][k+kk] = *fdata;
                }
	        }
	    }
	}
	for(int i=3; i<8; i++) {
	    for(int j=0; j<224; j++) {
	        for(int k=0; k<224; k++){
	            adata[i][j][k] = (float)0;
	        }
	    }
	}
	for(int j=0; j<224; j++) {
	    for(int k=0; k<224; k++){
	        for(int i=0; i<8; i++) {
	            float fdata2 = adata[i][j][k];
                int *idata = (int*)&fdata2;
	            tmp.range((i+1)*32-1, i*32) = *idata;
	        }
            m_fm[(j*224+k)] = tmp;
	    }
	}
}

template<typename wb_t, typename data_t, int num, int row, int col, int tf_row, int tf_col>
void reorder_output(data_t *m , wb_t *m_fm, int addr) {
   static data_t data[num][row][col];
   for(int ii=0; ii<num; ii+=UNROLL) {
       for(int r=0; r<row; r++) {
           for(int c=0; c<col; c++) {
               for(int i=0; i<UNROLL; i++) {
                   int idata = m_fm[addr/UNROLL +(r*col+c) + (ii/UNROLL)*row*col].range((i+1)*32-1, i*32);
                   float *fdata = (float*)&idata;
                   data[ii+i][r][c] = *fdata;
               }
           }
       }
   }
   for(int i=0; i<num; i++) {
       for(int j=0; j<row; j++) {
           for(int k=0; k<col; k++) {
               m[i*row*col + j*col + k] = data[i][j][k];
           }
       }
   }
}
void bwmemcpy(bw_t *dst, int addrdst, bw_t *src, int addrsrc, int length) {
    for(int i=0; i<length; i++) {
        dst[i + addrdst] = src[i + addrsrc]; 
    }
}
int main(int argc, char** argv)
{
    //-----------------------------
    //init DRAM
    data_t *DRAM_sw;
    int swDramSize = (WEIGHTSIZE+ INFM+ FM11);
    printf("INFM: %d, FM11: %d, Layer11: %d\n", INFM, FM11, Layer11);
    DRAM_sw = (data_t*) malloc(swDramSize*sizeof(data_t));
    if (DRAM_sw==NULL) {
        printf("malloc failure\n");
        exit (1);
    }
    //memset_int<data_t>(DRAM_sw, (data_t)0.0010, 0, swDramSize);
    memset_int<data_t>(DRAM_sw, (data_t)0.0000, WEIGHTSIZE, 224*100);
    memset_int<data_t>(DRAM_sw, (data_t)0.0000, (WEIGHTSIZE+224*100), (224*124+224*324));
    memset_int<data_t>(DRAM_sw, (data_t)0.0000, (WEIGHTSIZE+(224*224+224*324)), 124*224);
    //initialize weight
    memset_int<data_t>(DRAM_sw, (data_t)0.1000, 0, Layer11);
    memset_int<data_t>(DRAM_sw, (data_t)1.0000, Layer11, Layer12-Layer11);
    memset_int<data_t>(DRAM_sw, (data_t)0.40, Layer12, Layer21-Layer12);
    memset_int<data_t>(DRAM_sw, (data_t)0.50, Layer21, Layer22-Layer21);
    memset_int<data_t>(DRAM_sw, (data_t)0.80, Layer22, Layer31-Layer22);
    memset_int<data_t>(DRAM_sw, (data_t)0.80, Layer31, Layer32-Layer31);
    memset_int<data_t>(DRAM_sw, (data_t)0.80, Layer32, Layer33-Layer32);
    memset_int<data_t>(DRAM_sw, (data_t)0.80, Layer33, Layer41-Layer33);
    memset_int<data_t>(DRAM_sw, (data_t)0.80, Layer41, Layer42-Layer41);
    memset_int<data_t>(DRAM_sw, (data_t)0.80, Layer42, Layer43-Layer42);
    memset_int<data_t>(DRAM_sw, (data_t)0.80, Layer43, Layer51-Layer43);
    memset_int<data_t>(DRAM_sw, (data_t)0.80, Layer51, Layer52-Layer51);
    memset_int<data_t>(DRAM_sw, (data_t)0.80, Layer52, Layer53-Layer52);
    //initialize bias
    memset_int<data_t>(DRAM_sw, (data_t)1.0000, DATA*CONV11*KSIZE*KSIZE, CONV11);
    memset_int<data_t>(DRAM_sw, (data_t)0.0000, Layer12-CONV12, CONV12);
    memset_int<data_t>(DRAM_sw, (data_t)0.0000, Layer21-CONV21, CONV21);
    memset_int<data_t>(DRAM_sw, (data_t)0.0000, Layer22-CONV22, CONV22);
    memset_int<data_t>(DRAM_sw, (data_t)0.0000, Layer31-CONV31, CONV31);
    memset_int<data_t>(DRAM_sw, (data_t)0.0000, Layer32-CONV32, CONV32);
    memset_int<data_t>(DRAM_sw, (data_t)0.0000, Layer33-CONV33, CONV33);
    memset_int<data_t>(DRAM_sw, (data_t)0.0000, Layer41-CONV41, CONV41);
    memset_int<data_t>(DRAM_sw, (data_t)0.0000, Layer42-CONV42, CONV42);
    memset_int<data_t>(DRAM_sw, (data_t)1.0000, Layer43-CONV43, CONV43);
    memset_int<data_t>(DRAM_sw, (data_t)1.0000, Layer51-CONV51, CONV51);
    memset_int<data_t>(DRAM_sw, (data_t)1.0000, Layer52-CONV52, CONV52);
    memset_int<data_t>(DRAM_sw, (data_t)1.0000, Layer53-CONV53, CONV53);
    /*
    for(int i=0; i<64; i++) {
        for(int j=0; j<3; j++) {
            for(int k=0; k<9; k++) {
                if(j<8) DRAM_sw[i*3*9 + j*9 + k] = (10*(j+1) + k) + (data_t)(i+1)/100;
                else DRAM_sw[i*3*9 + j*9 + k] = (data_t)0;
            }
        } 
    } */
    printf("0. init DRAM_sw finish\n");

    int DramSize_weight = (WEIGHTSIZE); 
    data_t *DRAM_weight = (data_t*) malloc(DramSize_weight*sizeof(data_t));
    bw_t *DRAM_weight_reorder = (bw_t*) malloc(DramSize_weight*sizeof(data_t));
    int DramSize_featmap = (INFM + FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43+ FM51+FM52+FM53+PL53);
    bw_t *DRAM_featmap = (bw_t*) malloc(DramSize_featmap*sizeof(data_t));
    int DramSize_hw = LayerDef+ DramSize_weight + DramSize_featmap;
    bw_t *DRAM_hw = (bw_t*) malloc(DramSize_hw*sizeof(data_t));
    printf("1. init DRAM_hw finish\n");

    memcpy(DRAM_weight, DRAM_sw, sizeof(data_t)*WEIGHTSIZE);
    printf("2. memcpy DRAM_weight\n");
    prepare_weight(DRAM_weight_reorder, DRAM_weight); //reorder weights and transform them into bw_t
    printf("3. prepare DRAM weight\n");
    //--------------------------------------------
    //prepare image - memcpy DRAM_sw->DRAM_featmap and reorder first image
    prepare_image(DRAM_featmap, 0, DRAM_sw, WEIGHTSIZE, 224*224*3);
    printf("4. prepare DRAM feature map\n");

    layer lyr[13]={conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv3_3, conv4_1, conv4_2, conv4_3, conv5_1, conv5_2, conv5_3};
    int ly[13*16] = {
    lyr[0].fin, lyr[0].fout, lyr[0].finrow, lyr[0].fincol, lyr[0].frow, lyr[0].fcol, lyr[0].Ksize, lyr[0].Kstride, lyr[0].pad, lyr[0].mask, lyr[0].addr_in, lyr[0].addr_wght, lyr[0].addr_out, lyr[0].pool, lyr[0].relu, 0,
    lyr[1].fin, lyr[1].fout, lyr[1].finrow, lyr[1].fincol, lyr[1].frow, lyr[1].fcol, lyr[1].Ksize, lyr[1].Kstride, lyr[1].pad, lyr[1].mask, lyr[1].addr_in, lyr[1].addr_wght, lyr[1].addr_out, lyr[1].pool, lyr[1].relu, 0,
    lyr[2].fin, lyr[2].fout, lyr[2].finrow, lyr[2].fincol, lyr[2].frow, lyr[2].fcol, lyr[2].Ksize, lyr[2].Kstride, lyr[2].pad, lyr[2].mask, lyr[2].addr_in, lyr[2].addr_wght, lyr[2].addr_out, lyr[2].pool, lyr[2].relu, 0,
    lyr[3].fin, lyr[3].fout, lyr[3].finrow, lyr[3].fincol, lyr[3].frow, lyr[3].fcol, lyr[3].Ksize, lyr[3].Kstride, lyr[3].pad, lyr[3].mask, lyr[3].addr_in, lyr[3].addr_wght, lyr[3].addr_out, lyr[3].pool, lyr[3].relu, 0,
    lyr[4].fin, lyr[4].fout, lyr[4].finrow, lyr[4].fincol, lyr[4].frow, lyr[4].fcol, lyr[4].Ksize, lyr[4].Kstride, lyr[4].pad, lyr[4].mask, lyr[4].addr_in, lyr[4].addr_wght, lyr[4].addr_out, lyr[4].pool, lyr[4].relu, 0,
    lyr[5].fin, lyr[5].fout, lyr[5].finrow, lyr[5].fincol, lyr[5].frow, lyr[5].fcol, lyr[5].Ksize, lyr[5].Kstride, lyr[5].pad, lyr[5].mask, lyr[5].addr_in, lyr[5].addr_wght, lyr[5].addr_out, lyr[5].pool, lyr[5].relu, 0,
    lyr[6].fin, lyr[6].fout, lyr[6].finrow, lyr[6].fincol, lyr[6].frow, lyr[6].fcol, lyr[6].Ksize, lyr[6].Kstride, lyr[6].pad, lyr[6].mask, lyr[6].addr_in, lyr[6].addr_wght, lyr[6].addr_out, lyr[6].pool, lyr[6].relu, 0,
    lyr[7].fin, lyr[7].fout, lyr[7].finrow, lyr[7].fincol, lyr[7].frow, lyr[7].fcol, lyr[7].Ksize, lyr[7].Kstride, lyr[7].pad, lyr[7].mask, lyr[7].addr_in, lyr[7].addr_wght, lyr[7].addr_out, lyr[7].pool, lyr[7].relu, 0,
    lyr[8].fin, lyr[8].fout, lyr[8].finrow, lyr[8].fincol, lyr[8].frow, lyr[8].fcol, lyr[8].Ksize, lyr[8].Kstride, lyr[8].pad, lyr[8].mask, lyr[8].addr_in, lyr[8].addr_wght, lyr[8].addr_out, lyr[8].pool, lyr[8].relu, 0,
    lyr[9].fin, lyr[9].fout, lyr[9].finrow, lyr[9].fincol, lyr[9].frow, lyr[9].fcol, lyr[9].Ksize, lyr[9].Kstride, lyr[9].pad, lyr[9].mask, lyr[9].addr_in, lyr[9].addr_wght, lyr[9].addr_out, lyr[9].pool, lyr[9].relu, 0,
    lyr[10].fin, lyr[10].fout, lyr[10].finrow, lyr[10].fincol, lyr[10].frow, lyr[10].fcol, lyr[10].Ksize, lyr[10].Kstride, lyr[10].pad, lyr[10].mask, lyr[10].addr_in, lyr[10].addr_wght, lyr[10].addr_out, lyr[10].pool, lyr[10].relu, 0,
    lyr[11].fin, lyr[11].fout, lyr[11].finrow, lyr[11].fincol, lyr[11].frow, lyr[11].fcol, lyr[11].Ksize, lyr[11].Kstride, lyr[11].pad, lyr[11].mask, lyr[11].addr_in, lyr[11].addr_wght, lyr[11].addr_out, lyr[11].pool, lyr[11].relu, 0,
    lyr[12].fin, lyr[12].fout, lyr[12].finrow, lyr[12].fincol, lyr[12].frow, lyr[12].fcol, lyr[12].Ksize, lyr[12].Kstride, lyr[12].pad, lyr[12].mask, lyr[12].addr_in, lyr[12].addr_wght, lyr[12].addr_out, lyr[12].pool, lyr[12].relu, 0
    };
    printf("5. prepare layer definition\n");
    bw_t *DRAM_ly = (bw_t*) malloc(LayerDef*sizeof(int));
    for(int a=0; a<13; a++) {
        bw_t tmp=0;
        for(int b=0; b<16; b++){
           tmp.range(32*(b+1)-1, 32*b) = ly[a*16+b]; 
        } 
        DRAM_ly[a] = tmp; 
    }

    bwmemcpy(DRAM_hw, (WEIGHTSIZE+LayerDef)/UNROLL, DRAM_featmap, 0, DramSize_featmap/UNROLL); 
    printf("6. init device memory of feature map\n");
    bwmemcpy(DRAM_hw, (LayerDef)/UNROLL, DRAM_weight_reorder, 0, WEIGHTSIZE/UNROLL); 
    printf("7. init device memory of weight\n");
    bwmemcpy(DRAM_hw, 0, DRAM_ly, 0, 13); 
    printf("8. init device memory of weight\n");

    //write reordered data into data
    /*
    bw_t bw_data;
    ofstream ofresult("input_reorder.txt", ios::app);
    for(int i=0; i<224*224*8; i++) {
        if(i%8==0) bw_data = DRAM_featmap[i/8];
        int tt = bw_data.range((i%8+1)*32-1, (i%8)*32);
        float *data = (float*)&tt;
        //ofresult << *data <<"\t" << tt << "\t" << bw_data.range((i%8+1)*32-1, (i%8)*32) <<endl; 
        ofresult << *data <<endl; 
    }
    ofresult.close();
    */

    //-----------------------------
    //start compute
    vgg16_sw2(DRAM_sw);
    printf("software simulation finish!\n");
    vgg16(DRAM_hw, 13);
    printf("hardware simulation finish!\n");

    //-----------------------------
    //compare input reorder
    //for layer1 only
/*    
    float data1, data2, dataaccu=0.0;
    FILE *fp1=fopen("reorder_weight.txt", "r");
    FILE *fp2=fopen("load_weight.txt", "r");
    ofstream ofcompare("compare.txt", ios::app);
    for(int i=0; i<64*8*9; i++) {
        fscanf(fp1, "%f\n", &data1); 
        fscanf(fp2, "%f\n", &data2); 
        float tmodata = (data1-data2>0)?(data1-data2):(data2-data1);
        if(tmodata ==0) ofcompare << data1 << "\t\t" << data2 << "\t\t" << tmodata << endl;
        else  ofcompare << data1 << "\t\t" << data2 << "\t\t" << tmodata << "<<" << endl;
        dataaccu += tmodata;
    }
    printf("accumulate: %f\n", dataaccu);
    ofcompare.close();
    fclose(fp1);
    fclose(fp2); 
*/
    int cnt=0;
    //-----------------------------
    //compare result


#if 0
//layer[1]
    data_t tmp_sw=0;
    data_t tmp_hw=0;
    data_t sum=0;
    ofstream final("final.txt", ios::app);
    data_t *DRAM_result = (data_t*)malloc(FM11*sizeof(data_t));
    int result_addr = LayerDef + WEIGHTSIZE + INFM; 
    reorder_output<bw_t, data_t, CONV11, CONV11_R, CONV11_R, HWFR, HWFC>(DRAM_result, DRAM_hw, result_addr);
    for(int i=0; i<FM11; i++) {
        tmp_sw = DRAM_sw[WEIGHTSIZE+ INFM+ i];
        tmp_hw = DRAM_result[i];
        if (cmpratio(tmp_sw, tmp_hw)>2E-1) {
            //printf("sw:%f,\t hw:%f,\t ratio:%f\n", (float)tmp_sw, (float)tmp_hw, (float)cmpratio(tmp_sw, tmp_hw));
            final << "feature map:" << i/(CONV11_R*CONV11_R) << "\t\tsw: " <<  (float)tmp_sw << "\thw: " << (float)tmp_hw << "\t" << (float)cmpratio(tmp_sw, tmp_hw)<< endl;
            cnt+=1;
        }
        //final << "feature map:" << i/(CONV11_R*CONV11_R) << "\t\tsw: " <<  (float)tmp_sw << "\thw: " << (float)tmp_hw << "\t" << (float)cmpratio(tmp_sw, tmp_hw)<< endl;
    }
    final.close();
#endif
#if 0
//layer[1~2]
    int fmcnt=111;
    ofstream final("final.txt", ios::app);
    data_t tmp_sw=0;
    data_t tmp_hw=0;
    data_t sum=0;
    data_t *DRAM_result = (data_t*)malloc(PL12*sizeof(data_t));
    int result_addr = LayerDef + WEIGHTSIZE + INFM+ FM11+FM12; 

    reorder_output<bw_t, data_t, POOL12, POOL12_R, POOL12_R, HWFR, HWFC>(DRAM_result, DRAM_hw, result_addr);
    for(int i=0; i<PL12; i++) {
        tmp_sw = DRAM_sw[WEIGHTSIZE+ INFM+ i];
        tmp_hw = DRAM_result[i];
        if (cmpratio(tmp_sw, tmp_hw)>1E-5) {
            //printf("sw:%f,\t hw:%f,\t ratio:%f\n", (float)tmp_sw, (float)tmp_hw, (float)cmpratio(tmp_sw, tmp_hw));
            cnt+=1;
            final << "\t\tsw: " <<  (float)tmp_sw << "\thw: " << (float)tmp_hw << "\tratio:" << (float)cmpratio(tmp_sw, tmp_hw)<< endl;
        }
        //if(i/(POOL12_R*POOL12_R) != fmcnt) {
        //    fmcnt = i/(POOL12_R*POOL12_R);
        //    final << "feature map:" << fmcnt << endl;
        //}
    }
    final.close();
#endif
#if 0
//layer[1~2] without pool
    int fmcnt=111;
    ofstream final("final.txt", ios::app);
    data_t tmp_sw=0;
    data_t tmp_hw=0;
    data_t sum=0;
    data_t *DRAM_result = (data_t*)malloc(FM12*sizeof(data_t));
    int result_addr = LayerDef + WEIGHTSIZE + INFM+ FM11; 

    reorder_output<bw_t, data_t, CONV12, CONV12_R, CONV12_R, HWFR, HWFC>(DRAM_result, DRAM_hw, result_addr);
    for(int i=0; i<FM12; i++) {
        tmp_sw = DRAM_sw[WEIGHTSIZE+ INFM+ i];
        tmp_hw = DRAM_result[i];
        if (cmpratio(tmp_sw, tmp_hw)>1E-5) {
            //printf("sw:%f,\t hw:%f,\t ratio:%f\n", (float)tmp_sw, (float)tmp_hw, (float)cmpratio(tmp_sw, tmp_hw));
            cnt+=1;
        }
        if(i/(CONV12_R*CONV12_R) != fmcnt) {
            fmcnt = i/(CONV12_R*CONV12_R);
            final << "feature map:" << fmcnt << endl;
        }
        final << "\t\tsw: " <<  (float)tmp_sw << "\thw: " << (float)tmp_hw << "\tratio:" << (float)cmpratio(tmp_sw, tmp_hw)<< endl;
    }
    final.close();
#endif
#if 0
//layer[1~3]
    data_t tmp_sw=0;
    data_t tmp_hw=0;
    data_t sum=0;
    data_t *DRAM_result = (data_t*)malloc(PL33*sizeof(data_t));
    int result_addr = LayerDef + WEIGHTSIZE + INFM+ FM11+FM12+PL12+ FM21+FM22+PL22+ FM31+FM32+FM33; 

    reorder_output<bw_t, data_t, POOL33, POOL33_R, POOL33_R, HWFR, HWFC>(DRAM_result, DRAM_hw, result_addr);
    for(int i=0; i<PL33; i++) {
        tmp_sw = DRAM_sw[WEIGHTSIZE+ INFM+ i];
        tmp_hw = DRAM_result[i];
        //if (cmpratio(tmp_sw, tmp_hw)>1E-4) {
            printf("sw:%f,\t hw:%f,\t ratio:%f\n", (float)tmp_sw, (float)tmp_hw, (float)cmpratio(tmp_sw, tmp_hw));
        //    cnt+=1;
        //}
    } 
#endif
#if 1
//layer[1~5]
    data_t tmp_sw=0;
    data_t tmp_hw=0;
    data_t sum=0;
    data_t *DRAM_result = (data_t*)malloc(PL53*sizeof(data_t));
    int result_addr = LayerDef + WEIGHTSIZE + INFM+ FM11+FM12+PL12+ FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43+ FM51+FM52+FM53; 

    reorder_output<bw_t, data_t, POOL53, POOL53_R, POOL53_R, HWFR, HWFC>(DRAM_result, DRAM_hw, result_addr);

    ofstream final("final.txt", ios::app);
    for(int i=0; i<PL53; i++) {
        tmp_sw = DRAM_sw[WEIGHTSIZE+ INFM+ i];
        tmp_hw = DRAM_result[i];
        if (cmpratio(tmp_sw, tmp_hw)>1E-5) {
            printf("sw:%f,\t hw:%f,\t ratio:%f\n", (float)tmp_sw, (float)tmp_hw, (float)cmpratio(tmp_sw, tmp_hw));
            cnt+=1;
        }
        final << "\t\tsw: " <<  (float)tmp_sw << "\thw: " << (float)tmp_hw << "\tratio:" << (float)cmpratio(tmp_sw, tmp_hw)<< endl;
    } 
    final.close();
    printf("sizeof(float): %d ,\n", sizeof(float));
#endif


    free(DRAM_result);
    free(DRAM_weight);
    free(DRAM_featmap);
    free(DRAM_sw);
    free(DRAM_hw);

    if(cnt==0) {
        printf("SUCCESS\n");
    }
    else {
        printf("FAIL\n");
        printf("cnt:%d\n", cnt);
    }


  // Validate our results
/*  timeval startSw, endSw;
  gettimeofday(&startSw, NULL);

  gettimeofday(&endSw, NULL);
  printf("software time :%8.6f ms\n ", (endSw.tv_sec-startSw.tv_sec)*1e+3 + (endSw.tv_usec-startSw.tv_usec)*1e-03 );
  printf("Compute Golden Result Finish\n");
*/

}
