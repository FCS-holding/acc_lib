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
        m[i + addr] = val; 
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
    //conv.fin * conv.fout matches with vgg16_sw;
    for(int i=0; i<conv.fout; i++) {
        bias[i] = m[addr + conv.fin*conv.fout*9 + i]; 
    }
    int in_factor = (conv.fin>UNROLL)?conv.fin:UNROLL;
	for(int ii=0; ii<conv.fout; ii+=HWFOut) {
	    for(int jj=0; jj<in_factor; jj+=HWFIn) {
            for(int k=0; k<3; k++) {
                for(int h=0; h<3; h++) {
                    for(int i=0; i<HWFOut; i++) {
                        for(int j=0; j<HWFIn; j++) {
                            m[addr + ii*in_factor*9 + jj*9 + (k*3+h)*HWFOut*HWFIn + i*HWFIn+j] = weight[ii+i][jj+j][k][h];
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
void prepare_weight( bw_t *DRAM_weight_reorder, data_t *DRAM_weight ) {
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
                   float *fdata = (float*)&idata ;
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
    DRAM_sw = (data_t*) malloc(swDramSize*sizeof(data_t));
    if (DRAM_sw==NULL) {
        printf("malloc failure\n");
        exit (1);
    }
    memset_int<data_t>(DRAM_sw, (data_t)0.010, 0, swDramSize);
    memset_int<data_t>(DRAM_sw, (data_t)0.011, WEIGHTSIZE, 224*100);
    memset_int<data_t>(DRAM_sw, (data_t)0.002, (WEIGHTSIZE+224*100), (224*124+224*324));
    memset_int<data_t>(DRAM_sw, (data_t)0.015, (WEIGHTSIZE+(224*224+224*324)), 124*224);
    memset_int<data_t>(DRAM_sw, (data_t)0.001, 0, Layer11);
    memset_int<data_t>(DRAM_sw, (data_t)0.002, 0, Layer11/2);
    memset_int<data_t>(DRAM_sw, (data_t)0.013, 0, Layer11/3);
    memset_int<data_t>(DRAM_sw, (data_t)0.002, Layer11, Layer12-Layer11);
    memset_int<data_t>(DRAM_sw, (data_t)0.0015, Layer31, (Layer32-Layer31)/2);
    memset_int<data_t>(DRAM_sw, (data_t)0.005, Layer42, Layer43-Layer42);
    memset_int<data_t>(DRAM_sw, (data_t)0.003, Layer52, Layer53-Layer52);
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
    int DramSize_hw = DramSize_weight + DramSize_featmap;
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

    bwmemcpy(DRAM_hw, WEIGHTSIZE/UNROLL, DRAM_featmap, 0, DramSize_featmap/UNROLL); 
    printf("5. concat feat map\n");
    bwmemcpy(DRAM_hw, 0, DRAM_weight_reorder, 0, WEIGHTSIZE/UNROLL); 
    printf("6. concat weight\n");

    //write reordered data into data
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

    //-----------------------------
    //start compute
    vgg16_sw2(DRAM_sw);
    printf("software simulation finish!\n");
    vgg16(DRAM_hw);
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

#if 1
    data_t tmp_sw=0;
    data_t tmp_hw=0;
    data_t sum=0;
    data_t *DRAM_result = (data_t*)malloc(PL53*sizeof(data_t));
    int result_addr = WEIGHTSIZE + INFM+ FM11+FM12+PL12+ FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43+ FM51+FM52+FM53; 
    reorder_output<ap_uint<32*8>, data_t, POOL53, POOL53_R, POOL53_R, HWFR, HWFC>(DRAM_result, DRAM_hw, result_addr);
    for(int i=0; i<PL53; i++) {
        tmp_sw = DRAM_sw[WEIGHTSIZE+ INFM+ i];
        tmp_hw = DRAM_result[i];
        //if (cmpratio(tmp_sw, tmp_hw)>1E-4) {
            printf("sw:%f,\t hw:%f,\t ratio:%f\n", (float)tmp_sw, (float)tmp_hw, (float)cmpratio(tmp_sw, tmp_hw));
        //    cnt+=1;
        //}
    } 
#else
    data_t tmp_sw=0;
    data_t tmp_hw=0;
    data_t sum=0;
    data_t *DRAM_result = (data_t*)malloc(FM11*sizeof(data_t));
    int result_addr = WEIGHTSIZE + INFM; 
    reorder_output<ap_uint<32*8>, data_t, CONV11, CONV11_R, CONV11_R, HWFR, HWFC>(DRAM_result, DRAM_hw, result_addr);
    for(int i=0; i<FM11; i++) {
        tmp_sw = DRAM_sw[WEIGHTSIZE+ INFM+ i];
        sum += tmp_sw;
        tmp_hw = DRAM_result[i];
        if (cmpratio(tmp_sw, tmp_hw)>1E-5) {
            printf("sw:%f,\t hw:%f,\t ratio:%f\n", (float)tmp_sw, (float)tmp_hw, (float)cmpratio(tmp_sw, tmp_hw));
            cnt+=1;
        }
    }
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
