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
                                data_in[(i+m)*image_width*image_height + (j+n)*image_width + l+k] * filter[m*filter_h*filter_w + n*filter_w + l];
                        }
                    }
                }
            }
        }
    }
}

float filter[FILTER_LENGTH];

int main(int argc, char * argv[]){
    #ifdef MCC_ACC
    __merlin_init(argv[1]);
    #endif
    float *data_in          = (float*)malloc(sizeof(float) * (DATA_IN_LENGTH));
    float *data_out_cpu     = (float*)malloc(sizeof(float) * (DATA_OUT_LENGTH));
    float *data_out_merlin  = (float*)malloc(sizeof(float) * (DATA_OUT_LENGTH));
    float *filter           = (float*)malloc(sizeof(float) * (FILTER_LENGTH));
    
    printf("init data_in\n");
    for(int i = 0; i < image_depth; i++){
        for(int j = 0; j < image_height; j++){
            for(int k = 0; k < image_width; k++){
                data_in[i*image_height*image_width + j*image_width + k] = rand()%10;
            }
        }
    }

    printf("init filter\n");
    for(int i = 0; i < filter_d; i++){
        for(int j = 0; j < filter_h; j++){                                  
            for(int k = 0; k < filter_w; k++){                               
                filter[i*filter_h*filter_w + j*filter_w + k] = rand()%10;    
            }                                                                   
        }                                                                       
    }                                                                           
   
    printf("FILTER_SIZE = %d*%d*%d\n", filter_w,filter_h,filter_d);
    printf("IMAGE_SIZE = %d*%d*%d\n", image_width,image_height,image_depth);


    printf("start execute kernel\n");
    #ifdef MCC_ACC
    __merlin_conv_3d_kernel(data_in, filter, data_out_merlin);
    #endif

    printf("start run cpu\n");
    conv_3d_cpu(data_in, filter, data_out_cpu);

    #ifdef MCC_ACC
    __merlin_release();
    #endif


    int test_pass_number = 0;
    int test_number = DATA_OUT_LENGTH;
    for(int i = 0; i < test_number; i++){
        //if((data_out_cpu[i] == data_out_kernel[i]) && (data_out_cpu[i] == data_out_merlin[i]) )
        if((data_out_cpu[i] == data_out_merlin[i]) )
            test_pass_number = test_pass_number + 1;
        else {
            test_pass_number = test_pass_number;
            //printf("%4d cpu loop :%15.6f ,",i, data_out_cpu[i]);
            //printf("kernel :%15.6f ,", data_out_kernel[i]);
            //printf("merlin :%15.6f\n", data_out_merlin[i]);
        }
    }

    if(test_pass_number == test_number)
        printf("Test Passed!\n");
    else
        printf("Test Failed!\n");

    return 0;
} 
