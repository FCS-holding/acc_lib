#include <string>
#include <math.h>
#include "vgg16.hpp"

int lines_of_bin(FILE* file) {
  if (file==NULL) {
    return 0;
  }
  fseek(file, 0, 2);  
  int lines = ftell(file)/sizeof(float);
  fseek(file, 0, 0);  
  return lines;
}

int lines_of_file(FILE* file) {
  if (file==NULL) {
    return 0;
  }
  int lines = 0;
  while(!feof(file)) {
    char ch = fgetc(file);
    if(ch == '\n') {
      lines++;
    }
  }
  rewind(file);
  return lines;
}

template<typename data_t>
void init_dram(data_t* m) {
  FILE* fp;
  FILE* list = fopen("/curr/chenzhang/MLlib/vgg/vgg_weight_bin/LIST", "r");
  int list_lines = lines_of_file(list); 
  //printf("%d lines of list\n", list_lines);
  //int m_lines = 0;
  char filename[220]; 
  int laddr[26] = { 0, DATA*CONV11*KSIZE*KSIZE, Layer11, Layer12-CONV12, \
                    Layer12, Layer21-CONV21, Layer21, Layer22-CONV22, \
                    Layer22, Layer31-CONV31, Layer31, Layer32-CONV32, Layer32, Layer33-CONV33, \
                    Layer33, Layer41-CONV41, Layer41, Layer42-CONV42, Layer42, Layer43-CONV43, \
                    Layer43, Layer51-CONV51, Layer51, Layer52-CONV52, Layer52, Layer53-CONV53};
  float *buffer;
  for(int i=0; i<list_lines; i++) {
    fscanf(list, "%s\n", filename);       //get file name
    fp = fopen(filename, "rb");           //open file
    int bin_lines = lines_of_bin(fp);     //get file lines
    buffer = (float*)malloc(sizeof(data_t)*bin_lines);
    fread(buffer, sizeof(data_t), bin_lines, fp);
    for(int j=0; j<bin_lines; j++) {
    //  m[m_lines + j] = buffer[j];
      m[laddr[i] + j] = buffer[j];
    }
    //m_lines += bin_lines;
    free(buffer);
    fclose(fp);
  }
  rewind(list);
  /*/verfiy process
  m_lines = 0;
  float *verify;
  int cnt=0;
  for(int i=0; i<list_lines; i++) {
    fscanf(list, "%s\n", filename);       //get file name
    //printf("file %d: %s\n", i, filename); //print file name
    fp = fopen(filename, "rb");           //open file
    int bin_lines = lines_of_bin(fp);     //get file lines
    verify = (float*) malloc(sizeof(data_t)*bin_lines);
    fread(verify, sizeof(data_t), bin_lines, fp);
    for(int j=0; j<bin_lines; j++) {
      if(m[m_lines+j]!=verify[j]) cnt+=1; 
    }
    m_lines += bin_lines;
    free(verify);
    fclose(fp);
  }
  if(cnt!=0) printf("verify NOT pass, %d errors\n", cnt);
  else printf("verify pass!\n"); */
  fclose(list);
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

void prepare_weight(data_t *DRAM_weight ) {
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
}

void prepare_image(data_t *m, int addr_m) {

	float adata[UNROLL][224][224];
	for(int i=0; i<3; i++) {
	    for(int j=0; j<224; j++) {
	        for(int k=0; k<224; k++){
	                adata[i][j][k] = m[i*224*224+j*224+k + addr_m];
	        }
	    }
	}
	for(int i=3; i<UNROLL; i++) {
	    for(int j=0; j<224; j++) {
	        for(int k=0; k<224; k++){
	            adata[i][j][k] = (data_t)0;
	        }
	    }
	}
	for(int j=0; j<224; j++) {
	    for(int k=0; k<224; k++){
	        for(int i=0; i<UNROLL; i++) {
	            m[(j*224+k)*UNROLL + i + addr_m] = adata[i][j][k];
	        }
	    }
	}
}

void reorder(data_t *fdram) {
                                    
    prepare_weight(fdram); //reorder weights and transform them into bw_t
    prepare_image(fdram, WEIGHTSIZE);

}

template<typename data_t, int num, int row, int col, int tf_row, int tf_col>
void reorder_output(data_t *m) {
   static data_t data[num][row][col];
   for(int ii=0; ii<num; ii+=UNROLL) {
       for(int r=0; r<row; r++) {
           for(int c=0; c<col; c++) {
               for(int i=0; i<UNROLL; i++) {
                   data[ii+i][r][c] = m[(r*col+c)*UNROLL + (ii)*row*col + i];
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

template<typename data_t, int FIN, int FOUT, int Kernel>
void init_weights(data_t *m, int addr, data_t *weights, data_t *bias) {
    for(int i=0; i<FOUT; i++){
        bias[i] = m[addr+ FIN*FOUT*Kernel*Kernel + i];
        for(int j=0; j<FIN; j++){
            for(int k=0; k<Kernel; k++) {
                for(int s=0; s<Kernel; s++) {
                    weights[i*FIN*Kernel*Kernel + j*Kernel*Kernel + k*Kernel + s] = m[addr + i*FIN*Kernel*Kernel + j*Kernel*Kernel + k*Kernel + s];
    }   }   }   }
}

template<typename data_t, int FIN_, int FOUT_, int FINR_, int FINC_, int FOUTR_, int FOUTC_, int KSIZE_, int KSTRIDE_>
void conv_sw(data_t *in, data_t *Cout, data_t *weight, data_t *bias)
{
    //---conv---//
    int Couti, Coutj, Coutk;
    int Cini;
    int Cwi, Cwj;

    for(Coutj=0; Coutj<FOUTR_; Coutj++){
        for(Coutk=0; Coutk<FOUTC_; Coutk++){
            for(Couti=0; Couti<FOUT_; Couti++){
                Cout[Couti*FOUTR_*FOUTC_ +Coutj*FOUTC_ +Coutk] = (data_t)bias[Couti];
                for(Cini=0; Cini<FIN_; Cini++){
                    for(Cwi=0; Cwi<KSIZE_; Cwi++){
                        for(Cwj=0; Cwj<KSIZE_; Cwj++){
                            Cout[Couti*FOUTR_*FOUTC_ +Coutj*FOUTC_ +Coutk] +=
                            in[Cini*FINR_*FINC_ +(Coutj*KSTRIDE_+Cwi)*FINC_ +(Coutk*KSTRIDE_+Cwj)] * weight[Couti*FIN_*KSIZE_*KSIZE_ +Cini*KSIZE_*KSIZE_+ Cwi*KSIZE_ + Cwj];
    }   }   }   }   }   }
}
template<typename data_t, int FOUT_, int FOUTR_, int FOUTC_>
void relu_sw(data_t *Pout)
{
    //---relu---//
    for(int i=0; i<FOUT_*FOUTR_*FOUTC_; i++){
        if(Pout[i] < (data_t)0)
            Pout[i] = (data_t)0;
    }
}
template<typename data_t, int FOUT_, int FINR_, int FINC_, int FOUTR_, int FOUTC_, int KSIZE_, int KSTRIDE_>
void pool_sw( data_t *Cout, data_t *Pout)
{
    //---pool---//
    int Pouti, Poutj, Poutk;
    int Pini, Pinj;
    for(Pouti=0; Pouti<FOUT_; Pouti++){
        for(Poutj=0; Poutj<FOUTR_; Poutj++){
            for(Poutk=0; Poutk<FOUTC_; Poutk++){
                Pout[Pouti*FOUTR_*FOUTC_ + Poutj*FOUTC_ + Poutk] = Cout[Pouti*FINR_*FINC_ + Poutj*KSTRIDE_*FINC_ + Poutk*KSTRIDE_];
                for(Pini=0; Pini<KSIZE_; Pini++){
                    for(Pinj=0; Pinj<KSIZE_; Pinj++){
                        if(Cout[Pouti*FINC_*FINR_+ (Poutj*KSTRIDE_+Pini)*FINC_ + (Poutk*KSTRIDE_+Pinj)] > Pout[Pouti*FOUTR_*FOUTC_ + Poutj*FOUTC_ + Poutk])
                            Pout[Pouti*FOUTR_*FOUTC_ + Poutj*FOUTC_ + Poutk] = Cout[Pouti*FINC_*FINR_+ (Poutj*KSTRIDE_+Pini)*FINC_ + (Poutk*KSTRIDE_+Pinj)];
     }   }   }   }   }
}
template<typename data_t, int FOUT_, int FIN_, int FH_> //FH_=1 for vectro multiplication
void mmult(data_t out[FOUT_*FH_], data_t w[FOUT_*FIN_], data_t in[FIN_*FH_], data_t bias[FOUT_]) {
    for(int i=0; i<FOUT_; i++) {
        for(int j=0; j<FH_; j++) {
	    out[i*FH_+j] = bias[i];
            for(int k=0; k<FIN_; k++) {
                //TODO: try to make "in" col first to improve
                out[i*FH_+j] += w[i*FIN_+k]* in[k*FH_+j];
            }
        }
    }
}
template<typename data_t>
void softmax(data_t *out, int length) {
    data_t acc=0;
    for(int i=0; i<length; i++) {
       out[i] = exp(out[i]);  
       acc += out[i];
    }
    for(int i=0; i<length; i++) {
       out[i] = out[i]/acc;  
    }
}

template<typename data_t>
void read_ann_weight(char* filename, data_t* data) {
    FILE *fp;
    int bin_lines;
    fp = fopen(filename, "rb");
    bin_lines = lines_of_bin(fp); 
    fread(data, sizeof(data_t), bin_lines, fp);
    fclose(fp);
}

void ann(float *feat, float *result) {
    float* ann1 = (float*) malloc(sizeof(float)*4096);
    float* ann2 = (float*) malloc(sizeof(float)*4096);
    float* w1 = (float*) malloc(sizeof(float)*4096*7*7*512);
    float* bias = (float*) malloc(sizeof(float)*4096);

    char filename1_1[220] = "/curr/chenzhang/MLlib/vgg/vgg_weight_bin/fc6_wght.bin";
    char filename1_2[220] = "/curr/chenzhang/MLlib/vgg/vgg_weight_bin/fc6_bias.bin";
    char filename2_1[220] = "/curr/chenzhang/MLlib/vgg/vgg_weight_bin/fc7_wght.bin";
    char filename2_2[220] = "/curr/chenzhang/MLlib/vgg/vgg_weight_bin/fc7_bias.bin";
    char filename3_1[220] = "/curr/chenzhang/MLlib/vgg/vgg_weight_bin/fc8_wght.bin";
    char filename3_2[220] = "/curr/chenzhang/MLlib/vgg/vgg_weight_bin/fc8_bias.bin";

    read_ann_weight<float>(filename1_1, w1);
    read_ann_weight<float>(filename1_2, bias);
    mmult<float, 4096, 7*7*512, 1>(ann1, w1, feat, bias);
    relu_sw<float, 4096, 1, 1>(ann1);

    read_ann_weight<float>(filename2_1, w1);
    read_ann_weight<float>(filename2_2, bias);
    mmult<float, 4096, 4096, 1>(ann2, w1, ann1, bias);
    relu_sw<float, 4096, 1, 1>(ann2);

    read_ann_weight<float>(filename3_1, w1);
    read_ann_weight<float>(filename3_2, bias);
    mmult<float, 1000, 4096, 1>(result, w1, ann2, bias);
    softmax<float>(result, 1000);

    free(bias);
    free(w1);
    free(ann1);
    free(ann2);
}

template<typename data_t, int FOUT, int FR, int FC>
void dram_op(data_t *m, int addr, data_t *in, int pad, int F)
{
    for(int i=0; i<FOUT; i++) {
        for(int j=0; j<FR; j++) {
            for(int k=0; k<FC; k++) {
                if(F==READ) {
                    if((k>=pad)&&(j>=pad)&&(k<FC-pad)&&(j<FR-pad))
                        in[i*FR*FC+ j*FC+ k] = m[addr+ i*(FR-2*pad)*(FC-2*pad)+ (j-pad)*(FC-2*pad) + (k-pad)];
                    else in[i*FR*FC+ j*FC+ k] = (data_t) 0;
                }
                else { 
                    m[addr+ i*FR*FC+ j*FC+ k] = in[i*FR*FC+ j*FC+ k];
                }
            }
        }
    }
}
template<typename data_t, int FOUT_, int FINR_, int FINC_, int FOUTR_, int FOUTC_>
void transfer(data_t *in, data_t *Cout)
{
    for(int i=0; i<FOUT_; i++) {
        for(int j=0; j<FINR_; j++) {
            for(int k=0; k<FINC_; k++) {
                if((j>=1)&&(k>=1)&&(j<FINR_-1)&&(k<FINC_-1))
                    in[i*FINR_*FINC_ +j*FINC_ +k] = Cout[i*FOUTR_*FOUTC_+ (j-1)*FOUTC_+ (k-1)];
                else
                    in[i*FINR_*FINC_ +j*FINC_ +k] = (data_t) 0;
    }   }   }
}

void vgg16_sw2(data_t *m) {

static data_t Cin[226*226*64];
static data_t Cout[226*226*64];
static data_t Pout[226*226*64];
static data_t Weight[512*512*3*3];
static data_t bias[512];

dram_op<data_t, DATA, CONV11_I, CONV11_I>(m, WEIGHTSIZE, Cin, PAD, READ);
//---layer1_1----
init_weights<data_t, DATA, CONV11, KSIZE>(m, 0, Weight, bias);
conv_sw<data_t, DATA, CONV11, CONV11_I, CONV11_I, CONV11_R, CONV11_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV11, CONV11_R, CONV11_R>(Cout);
//dram_op<data_t, CONV11, CONV11_R, CONV11_R>(m, WEIGHTSIZE+ INFM , Cout, PAD, WRITE);

//---layer1_2----
transfer<data_t, CONV11, CONV12_I, CONV12_I, CONV11_R, CONV11_R>(Cin, Cout);
init_weights<data_t, CONV11, CONV12, KSIZE>(m, Layer11, Weight, bias);
conv_sw<data_t, CONV11, CONV12, CONV12_I, CONV12_I, CONV12_R, CONV12_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV12, CONV12_R, CONV12_R>(Cout);
pool_sw<data_t, POOL12, CONV12_R, CONV12_R, POOL12_R, POOL12_R, 2, 2>(Cout, Pout);
//dram_op<data_t, POOL12, POOL12_R, POOL12_R>(m, (WEIGHTSIZE+ INFM), Pout, PAD, WRITE);

//---layer2_1----
transfer<data_t, POOL12, CONV21_I, CONV21_I, POOL12_R, POOL12_R>(Cin, Pout);
init_weights<data_t, CONV12, CONV21, KSIZE>(m, Layer12, Weight, bias);
conv_sw<data_t, POOL12, CONV21, CONV21_I, CONV21_I, CONV21_R, CONV21_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV21, CONV21_R, CONV21_R>(Cout);
//dram_op<data_t, CONV21, CONV21_R, CONV21_R>(m, (WEIGHTSIZE+ INFM), Cout, PAD, WRITE);

//---layer2_2----
transfer<data_t, CONV21, CONV22_I, CONV22_I, CONV21_R, CONV21_R>(Cin, Cout);
init_weights<data_t, CONV21, CONV22, KSIZE>(m, Layer21, Weight, bias);
conv_sw<data_t, CONV21, CONV22, CONV22_I, CONV22_I, CONV22_R, CONV22_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV22, CONV22_R, CONV22_R>(Cout);
pool_sw<data_t, POOL22, CONV22_R, CONV22_R, POOL22_R, POOL22_R, 2, 2>(Cout, Pout);
//dram_op<data_t, POOL22, POOL22_R, POOL22_R>(m, (WEIGHTSIZE+ INFM), Pout, PAD, WRITE);

//---layer3_1----
transfer<data_t, POOL22, CONV31_I, CONV31_I, POOL22_R, POOL22_R>(Cin, Pout);
init_weights<data_t, CONV22, CONV31, KSIZE>(m, Layer22, Weight, bias);
conv_sw<data_t, POOL22, CONV31, CONV31_I, CONV31_I, CONV31_R, CONV31_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV31, CONV31_R, CONV31_R>(Cout);
//dram_op<data_t, CONV31, CONV31_R, CONV31_R>(m, (WEIGHTSIZE+ DATA*CONV11_R*CONV11_R), Cout31, PAD, WRITE);
//---layer3_2----
transfer<data_t, CONV31, CONV32_I, CONV32_I, CONV31_R, CONV31_R>(Cin, Cout);
init_weights<data_t, CONV31, CONV32, KSIZE>(m, Layer31, Weight, bias);
conv_sw<data_t, CONV31, CONV32, CONV32_I, CONV32_I, CONV32_R, CONV32_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV32, CONV32_R, CONV32_R>(Cout);
//dram_op<data_t, CONV32, CONV32_R, CONV32_R>(m, (WEIGHTSIZE+ DATA*CONV11_R*CONV11_R), Cout32, PAD, WRITE);
//--- layer3_3----
transfer<data_t, CONV32, CONV33_I, CONV33_I, CONV32_R, CONV32_R>(Cin, Cout);
init_weights<data_t, CONV32, CONV33, KSIZE>(m, Layer32, Weight, bias);
conv_sw<data_t, CONV32, CONV33, CONV33_I, CONV33_I, CONV33_R, CONV33_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV33, CONV33_R, CONV33_R>(Cout);
pool_sw<data_t, POOL33, CONV33_R, CONV33_R, POOL33_R, POOL33_R, 2, 2>(Cout, Pout);
//dram_op<data_t, POOL33, POOL33_R, POOL33_R>(m, (WEIGHTSIZE+ INFM), Pout, PAD, WRITE);

//---layer4_1----
transfer<data_t, POOL33, CONV41_I, CONV41_I, POOL33_R, POOL33_R>(Cin, Pout);
init_weights<data_t, CONV33, CONV41, KSIZE>(m, Layer33, Weight, bias);
conv_sw<data_t, POOL33, CONV41, CONV41_I, CONV41_I, CONV41_R, CONV41_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV41, CONV41_R, CONV41_R>(Cout);
//dram_op<data_t, CONV41, CONV41_R, CONV41_R>(m, (WEIGHTSIZE+ DATA*CONV11_R*CONV11_R), Cout41, PAD, WRITE);
//---layer4_2----
transfer<data_t, CONV41, CONV42_I, CONV42_I, CONV41_R, CONV41_R>(Cin, Cout);
init_weights<data_t, CONV41, CONV42, KSIZE>(m, Layer41, Weight, bias);
conv_sw<data_t, CONV41, CONV42, CONV42_I, CONV42_I, CONV42_R, CONV42_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV42, CONV42_R, CONV42_R>(Cout);
//dram_op<data_t, CONV42, CONV42_R, CONV42_R>(m, (WEIGHTSIZE+ DATA*CONV11_R*CONV11_R), Cout42, PAD, WRITE);
//--- layer4_3----
transfer<data_t, CONV42, CONV43_I, CONV43_I, CONV42_R, CONV42_R>(Cin, Cout);
init_weights<data_t, CONV42, CONV43, KSIZE>(m, Layer42, Weight, bias);
conv_sw<data_t, CONV42, CONV43, CONV43_I, CONV43_I, CONV43_R, CONV43_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV43, CONV43_R, CONV43_R>(Cout);
pool_sw<data_t, POOL43, CONV43_R, CONV43_R, POOL43_R, POOL43_R, 2, 2>(Cout, Pout);
//dram_op<data_t, POOL43, POOL43_R, POOL43_R>(m, (WEIGHTSIZE+ INFM), Pout, PAD, WRITE);

//---layer5_1----
transfer<data_t, POOL43, CONV51_I, CONV51_I, POOL43_R, POOL43_R>(Cin, Pout);
init_weights<data_t, CONV43, CONV51, KSIZE>(m, Layer43, Weight, bias);
conv_sw<data_t, POOL43, CONV51, CONV51_I, CONV51_I, CONV51_R, CONV51_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV51, CONV51_R, CONV51_R>(Cout);
//dram_op<data_t, CONV51, CONV51_R, CONV51_R>(m, (WEIGHTSIZE+ DATA*CONV11_R*CONV11_R), Cout51, PAD, WRITE);
//---layer5_2----
transfer<data_t, CONV51, CONV52_I, CONV52_I, CONV51_R, CONV51_R>(Cin, Cout);
init_weights<data_t, CONV51, CONV52, KSIZE>(m, Layer51, Weight, bias);
conv_sw<data_t, CONV51, CONV52, CONV52_I, CONV52_I, CONV52_R, CONV52_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV52, CONV52_R, CONV52_R>(Cout);
//dram_op<data_t, CONV52, CONV52_R, CONV52_R>(m, (WEIGHTSIZE+ DATA*CONV11_R*CONV11_R), Cout52, PAD, WRITE);
//--- layer5_3----
transfer<data_t, CONV52, CONV53_I, CONV53_I, CONV52_R, CONV52_R>(Cin, Cout);
init_weights<data_t, CONV52, CONV53, KSIZE>(m, Layer52, Weight, bias);
conv_sw<data_t, CONV52, CONV53, CONV53_I, CONV53_I, CONV53_R, CONV53_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV53, CONV53_R, CONV53_R>(Cout);
pool_sw<data_t, POOL53, CONV53_R, CONV53_R, POOL53_R, POOL53_R, 2, 2>(Cout, Pout);
dram_op<data_t, POOL53, POOL53_R, POOL53_R>(m, (WEIGHTSIZE+ INFM), Pout, PAD, WRITE);

}



