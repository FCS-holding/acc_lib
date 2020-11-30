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

template<typename data_t, int FOUT_, int FOUTR_, int FOUTC_>
void relu_sw(data_t *Pout)
{
    //---relu---//
    for(int i=0; i<FOUT_*FOUTR_*FOUTC_; i++){
        if(Pout[i] < (data_t)0)
            Pout[i] = (data_t)0;
    }
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





