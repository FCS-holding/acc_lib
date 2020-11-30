#include <string>
#include <math.h>
#include "cnn_cfg.hpp"

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

template<typename d_type, int FOUT_, int FOUTR_, int FOUTC_>
void relu_sw(d_type *Pout)
{
    //---relu---//
    for(int i=0; i<FOUT_*FOUTR_*FOUTC_; i++){
        if(Pout[i] < (d_type)0)
            Pout[i] = (d_type)0;
    }
}

template<typename d_type, int FOUT_, int FIN_, int FH_> //FH_=1 for vectro multiplication
void mmult(d_type out[FOUT_*FH_], d_type w[FOUT_*FIN_], d_type in[FIN_*FH_], d_type bias[FOUT_]) {
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
template<typename d_type>
void softmax(d_type *out, int length) {
    d_type acc=0;
    for(int i=0; i<length; i++) {
       out[i] = exp(out[i]);  
       acc += out[i];
    }
    for(int i=0; i<length; i++) {
       out[i] = out[i]/acc;  
    }
}

template<typename d_type>
void read_ann_weight(char* filename, d_type* data) {
    FILE *fp;
    int bin_lines;
    fp = fopen(filename, "rb");
    bin_lines = lines_of_bin(fp); 
    fread(data, sizeof(d_type), bin_lines, fp);
    fclose(fp);
}

template<typename d_type>
void load_ann_model(float* w1, float* b1, float* w2, float* b2, float* w3, float* b3) {

  char filename1_1[220] = "/curr/chenzhang/MLlib/vgg/vgg_weight_bin/fc6_wght.bin";
  char filename1_2[220] = "/curr/chenzhang/MLlib/vgg/vgg_weight_bin/fc6_bias.bin";
  char filename2_1[220] = "/curr/chenzhang/MLlib/vgg/vgg_weight_bin/fc7_wght.bin";
  char filename2_2[220] = "/curr/chenzhang/MLlib/vgg/vgg_weight_bin/fc7_bias.bin";
  char filename3_1[220] = "/curr/chenzhang/MLlib/vgg/vgg_weight_bin/fc8_wght.bin";
  char filename3_2[220] = "/curr/chenzhang/MLlib/vgg/vgg_weight_bin/fc8_bias.bin";
  read_ann_weight<float>(filename1_1, w1);
  read_ann_weight<float>(filename1_2, b1);
  read_ann_weight<float>(filename2_1, w2);
  read_ann_weight<float>(filename2_2, b2);
  read_ann_weight<float>(filename3_1, w3);
  read_ann_weight<float>(filename3_2, b3);

}

void ann2(float *feat, float *result, float* w1, float* bias1, float* w2, float* bias2, float* w3, float* bias3) {
    float* ann1 = (float*) malloc(sizeof(float)*4096);
    float* ann2 = (float*) malloc(sizeof(float)*4096);
    //float* w2 = (float*) malloc(sizeof(float)*4096*4096);
    //float* bias2 = (float*) malloc(sizeof(float)*4096);
    //float* w1 = (float*) malloc(sizeof(float)*4096*7*7*512);
    //float* bias = (float*) malloc(sizeof(float)*4096);

    //char filename1_1[220] = "/curr/chenzhang/MLlib/vgg/vgg_weight_bin/fc6_wght.bin";
    //char filename1_2[220] = "/curr/chenzhang/MLlib/vgg/vgg_weight_bin/fc6_bias.bin";
    //char filename2_1[220] = "/curr/chenzhang/MLlib/vgg/vgg_weight_bin/fc7_wght.bin";
    //char filename2_2[220] = "/curr/chenzhang/MLlib/vgg/vgg_weight_bin/fc7_bias.bin";
    //char filename3_1[220] = "/curr/chenzhang/MLlib/vgg/vgg_weight_bin/fc8_wght.bin";
    //char filename3_2[220] = "/curr/chenzhang/MLlib/vgg/vgg_weight_bin/fc8_bias.bin";

    //struct timeval t0, t1, t2;
    //gettimeofday(&t0, NULL);
    //gettimeofday(&t1, NULL);
    //gettimeofday(&t2, NULL);
    //float read_wght = (t1.tv_sec-t0.tv_sec)*1e+3 + (t1.tv_usec-t0.tv_usec)*1e-03 ;
    //float time_over = (t2.tv_sec-t0.tv_sec)*1e+3 + (t2.tv_usec-t0.tv_usec)*1e-03 ;
    //printf("ann: read: %f, overall: %f,\n", read_wght, time_over);

    //read_ann_weight<float>(filename1_1, w1);
    //read_ann_weight<float>(filename1_2, bias);
    mmult<float, 4096, 7*7*512, 1>(ann1, w1, feat, bias1);
    relu_sw<float, 4096, 1, 1>(ann1);

    //read_ann_weight<float>(filename2_1, w2);
    //read_ann_weight<float>(filename2_2, bias2);
    mmult<float, 4096, 4096, 1>(ann2, w2, ann1, bias2);
    relu_sw<float, 4096, 1, 1>(ann2);

    //read_ann_weight<float>(filename3_1, w2);
    //read_ann_weight<float>(filename3_2, bias2);
    mmult<float, 1000, 4096, 1>(result, w3, ann2, bias3);
    softmax<float>(result, 1000);

    //free(bias2);
    //free(w2);
    free(ann1);
    free(ann2);
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

    //struct timeval t0, t1, t2;
    //gettimeofday(&t0, NULL);
    //gettimeofday(&t1, NULL);
    //gettimeofday(&t2, NULL);
    //float read_wght = (t1.tv_sec-t0.tv_sec)*1e+3 + (t1.tv_usec-t0.tv_usec)*1e-03 ;
    //float time_over = (t2.tv_sec-t0.tv_sec)*1e+3 + (t2.tv_usec-t0.tv_usec)*1e-03 ;
    //printf("ann: read: %f, overall: %f,\n", read_wght, time_over);

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



