#ifdef MCC_ACC
#include MCC_ACC_H_FILE
#endif

#include <time.h>
#include <stdio.h>
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "conv_3d.h"

void conv_3d_kernel(float data_in[], float filter[], float data_out[]);

void conv_3d_cpu(float data_in[DATA_IN_LENGTH], float filter[FILTER_LENGTH], float data_out[DATA_OUT_LENGTH]){
    int i,j,k,m,n,l;
    for(i = 0; i < image_depth - filter_d + 1; i++){
        for(j = 0; j < image_height - filter_h + 1; j++){
            for(k = 0; k < image_width - filter_w + 1; k++){
                for(m = 0; m < filter_d; m++){
                    for(n = 0; n < filter_h; n++){
                        for(l = 0; l < filter_w; l++){
                            data_out[i*(image_height - filter_h + 1)*(image_width - filter_w + 1) + j*(image_width - filter_w + 1) + k] += \
                                data_in[(i+m)*DATA_IN_LENGTH/image_depth + (j+n)*image_width + l+k] * filter[m*filter_h*filter_w + n*filter_w + l];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char * argv[]){
    #ifdef MCC_ACC
    __merlin_init(argv[1]);
    #endif
    float *data_in          = (float*)malloc(sizeof(float) * (DATA_IN_LENGTH));
    float *data_out_cpu     = (float*)malloc(sizeof(float) * (DATA_OUT_LENGTH));
    float *data_out_merlin  = (float*)malloc(sizeof(float) * (DATA_OUT_LENGTH));
    float filter[FILTER_LENGTH];
    printf("init data_in\n");
    for(int i = 0; i < image_depth; i++){
        for(int j = 0; j < DATA_IN_LENGTH/image_depth/image_width; j++){
            for(int k = 0; k < image_width; k++){
                float x = rand()%10;
                data_in[i*DATA_IN_LENGTH/image_depth + j*image_width + k] = x;
            }
        }
    }
    printf("init filter\n");
    for(int i = 0; i < filter_d; i++){
        for(int j = 0; j < filter_h; j++){                                  
            for(int k = 0; k < filter_w; k++){                    
                float a = rand()%10;
                filter[i*filter_h*filter_w + j*filter_w + k] = a;    
            }                                                                   
        }                                                                       
    }       
   
    printf("PE = %d\n", PE);
    printf("FILTER_SIZE = %d\n", FILTER_SIZE);
    printf("IMAGE_SIZE = %d\n", IMAGE_SIZE);
    printf("OUT_IMAGE_SIZE = %d\n", OUT_IMAGE_SIZE);
    printf("STEP = %d\n", STEP);

    printf("START_0 = %d\n", START_0);
    printf("END_0 = %d\n", END_0);
    printf("START_SIZE_0 = %d\n", START_SIZE_0);
    printf("IN_SIZE_0 = %d\n", IN_SIZE_0);
    printf("OUT_START_0 = %d\n", OUT_START_0);
    printf("OUT_END_0 = %d\n", OUT_END_0);
    printf("OUT_DEPTH_0 = %d\n", OUT_DEPTH_0);
    printf("OUT_SIZE_0 = %d\n", OUT_SIZE_0);

    #if PE>=2
    printf("START_1 = %d\n", START_1);
    printf("END_1 = %d\n", END_1);
    printf("START_SIZE_1 = %d\n", START_SIZE_1);
    printf("IN_SIZE_1 = %d\n", IN_SIZE_1);
    printf("OUT_START_1 = %d\n", OUT_START_1);
    printf("OUT_END_1 = %d\n", OUT_END_1);
    printf("OUT_DEPTH_1 = %d\n", OUT_DEPTH_1);
    printf("OUT_SIZE_1 = %d\n", OUT_SIZE_1);
    #endif
    
    #if PE>=3
    printf("START_2 = %d\n", START_2);
    printf("END_2 = %d\n", END_2);
    printf("START_SIZE_2 = %d\n", START_SIZE_2);
    printf("IN_SIZE_2 = %d\n", IN_SIZE_2);
    printf("OUT_START_2 = %d\n", OUT_START_2);
    printf("OUT_END_2 = %d\n", OUT_END_2);
    printf("OUT_DEPTH_2 = %d\n", OUT_DEPTH_2);
    printf("OUT_SIZE_2 = %d\n", OUT_SIZE_2);
    #endif
    #ifdef MCC_ACC
    printf("split data\n");
    float * data_in_merlin_0 = (float *)malloc(IN_SIZE_0*sizeof(float));
    #if PE>=2
    float * data_in_merlin_1 = (float *)malloc(IN_SIZE_1*sizeof(float));
    #endif
    #if PE>=3
    float * data_in_merlin_2 = (float *)malloc(IN_SIZE_2*sizeof(float));
    #endif
    float * data_out_merlin_0 = (float *)malloc(OUT_SIZE_0*sizeof(float));
    for(int i = 0; i < IN_SIZE_0; i++){
        data_in_merlin_0[i] = 0;
    }
    #if PE>=2
    float * data_out_merlin_1 = (float *)malloc(OUT_SIZE_1*sizeof(float));
    for(int i = 0; i < IN_SIZE_1; i++){
        data_in_merlin_1[i] = 0;
    }
    #endif
    #if PE>=3
    float * data_out_merlin_2 = (float *)malloc(OUT_SIZE_2*sizeof(float));
    for(int i = 0; i < IN_SIZE_2; i++){
        data_in_merlin_2[i] = 0;
    }
    #endif


    for(int i=0; i<IMAGE_SIZE; i++) {
        for(int j=0; j<IN_LENGTH_0; j++) {
            for(int k=0; k<IMAGE_SIZE; k++) {
                data_in_merlin_0[i*IMAGE_SIZE*IN_LENGTH_0 + j*IMAGE_SIZE + k] \
                    = data_in[i*DATA_IN_LENGTH/image_depth + j*IMAGE_SIZE + k];
                //printf("data_in_merlin_0 :%4d, %f\n",i*IMAGE_SIZE*IN_LENGTH_0 + j*IMAGE_SIZE + k, data_in_merlin_0[i*IMAGE_SIZE*IN_LENGTH_0 + j*IMAGE_SIZE + k]);
            }
        }
    }

    #if PE>=2
    for(int i=0; i<IMAGE_SIZE; i++) {
        for(int j=0; j<IN_LENGTH_1; j++) {
            for(int k=0; k<IMAGE_SIZE; k++) {
                data_in_merlin_1[i*IMAGE_SIZE*IN_LENGTH_1 + j*IMAGE_SIZE + k] \
                    = data_in[i*DATA_IN_LENGTH/image_depth + (j+START_1)*IMAGE_SIZE + k];
                //printf("data_in_merlin_1 :%4d, %f\n",i*IMAGE_SIZE*IN_LENGTH_1 + j*IMAGE_SIZE + k, data_in_merlin_1[i*IMAGE_SIZE*IN_LENGTH_1 + j*IMAGE_SIZE + k]);
            }
        }
    }
    #endif

    #if PE>=3
    for(int i=0; i<IMAGE_SIZE; i++) {
        for(int j=0; j<IN_LENGTH_2; j++) {
            for(int k=0; k<IMAGE_SIZE; k++) {
                data_in_merlin_2[i*IMAGE_SIZE*IN_LENGTH_2 + j*IMAGE_SIZE + k] \
                    = data_in[i*DATA_IN_LENGTH/image_depth + (j+START_2)*IMAGE_SIZE + k];
                //printf("data_in_merlin_2 :%4d, %f\n",i*IMAGE_SIZE*IN_LENGTH_2 + j*IMAGE_SIZE + k, data_in_merlin_2[i*IMAGE_SIZE*IN_LENGTH_2 + j*IMAGE_SIZE + k]);
            }
        }
    }
    #endif


    printf("start execute kernel\n");
    __merlin_new_task(
        data_in_merlin_0,
        #if PE>=2
        data_in_merlin_1,
        #endif
        #if PE>=3
        data_in_merlin_2,
        #endif
        data_out_merlin_0,
        #if PE>=2
        data_out_merlin_1,
        #endif
        #if PE>=3
        data_out_merlin_2,
        #endif
        #if PE>=2
        filter,
        #endif
        #if PE>=3
        filter,
        #endif
        filter
        );
    printf("start merge data\n");
    for(int i=0; i<OUT_IMAGE_SIZE; i++) {
        for(int j=0; j<OUT_IMAGE_SIZE; j++) {
            for(int k=0; k<OUT_IMAGE_SIZE; k++) {
                if(j < OUT_DEPTH_0) {
                    data_out_merlin[i*OUT_IMAGE_SIZE*OUT_IMAGE_SIZE + j*OUT_IMAGE_SIZE + k] \
                    = data_out_merlin_0[i*OUT_IMAGE_SIZE*STEP + j*OUT_IMAGE_SIZE + k];
                #if PE>=2
                } else if(j < OUT_DEPTH_0+OUT_DEPTH_1) {
                    data_out_merlin[i*OUT_IMAGE_SIZE*OUT_IMAGE_SIZE + j*OUT_IMAGE_SIZE + k] \
                    = data_out_merlin_1[i*OUT_IMAGE_SIZE*STEP + (j-OUT_DEPTH_0)*OUT_IMAGE_SIZE + k];
                #endif
                #if PE>=3
                } else if(j < OUT_DEPTH_0+OUT_DEPTH_1+OUT_DEPTH_2) {
                    data_out_merlin[i*OUT_IMAGE_SIZE*OUT_IMAGE_SIZE + j*OUT_IMAGE_SIZE + k] \
                    = data_out_merlin_2[i*OUT_IMAGE_SIZE*STEP + (j-OUT_DEPTH_0-OUT_DEPTH_1)*OUT_IMAGE_SIZE + k];
                #endif
                }
            }
        }
    }
    #endif

    printf("start run cpu\n");
    conv_3d_cpu(data_in, filter, data_out_cpu);

    int test_pass_number = 0;
    int test_number = DATA_OUT_LENGTH;
    for(int i = 0; i < test_number; i++){
        if((data_out_cpu[i] == data_out_merlin[i])) {
            test_pass_number = test_pass_number + 1;
        } else {
            test_pass_number = test_pass_number;
            //printf("%4d cpu loop :%15.6f ,",i, data_out_cpu[i]);
            //printf("merlin :%15.6f\n", data_out_merlin[i]);
        }
    }

    printf("start compare data\n");
    if(test_pass_number == test_number)
        printf("Test Passed!\n");
    else
        printf("Test Failed!\n");

    #ifdef MCC_ACC
    __merlin_release();
    #endif
    return 0;
} 
