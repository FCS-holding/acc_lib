#ifdef MCC_ACC
#include MCC_ACC_H_FILE
#endif
#include <iostream>
#include <time.h>
#include <stdio.h>
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "conv_3d.h"
#include <chrono>
#include <vector>
using namespace std;

void conv_3d_cpu_one_plane(float data_in[LENGTH_IN_TILE], float filter[FILTER_IN_LENGTH], float data_out[LENGTH_OUT_TILE]){
    int j,k,m,n,l;
    for(j = 0; j < OUT_IMAGE_SIZE; j++){
        for(k = 0; k < OUT_IMAGE_SIZE; k++){
            for(m = 0; m < FILTER_SIZE; m++){
                for(l = 0; l < FILTER_SIZE; l++){
                    for(n = 0; n < FILTER_SIZE; n++){
                        int index_out = j*OUT_IMAGE_SIZE + k;
                        int index_in = m*IMAGE_SIZE*IMAGE_SIZE + (j+n)*IMAGE_SIZE + l+k;
                        data_out[index_out] += \
                            data_in[index_in] * filter[m*FILTER_SIZE*FILTER_SIZE + n*FILTER_SIZE + l];
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
    float *filter           = (float*)malloc(sizeof(float) * (FILTER_IN_LENGTH));
    
    printf("init data_in and filter\n");
    for(int i = 0; i < IMAGE_SIZE; i++){
        for(int j = 0; j < IMAGE_SIZE; j++){
            for(int k = 0; k < IMAGE_SIZE; k++){
                data_in[i*IMAGE_SIZE*IMAGE_SIZE + j*IMAGE_SIZE + k] = rand()%10;
            }
        }
    }
    for(int i = 0; i < FILTER_SIZE; i++){
        for(int j = 0; j < FILTER_SIZE; j++){                                  
            for(int k = 0; k < FILTER_SIZE; k++){                               
                filter[i*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k] = rand()%10;    
            }                                                                   
        }                                                                       
    }                                                                           
    for(int i = 0; i < OUT_IMAGE_SIZE; i++){
        for(int j = 0; j < OUT_IMAGE_SIZE; j++){
            for(int k = 0; k < OUT_IMAGE_SIZE; k++){
                data_out_cpu[i*OUT_IMAGE_SIZE*OUT_IMAGE_SIZE + j*OUT_IMAGE_SIZE + k] = 0;
                data_out_merlin[i*OUT_IMAGE_SIZE*OUT_IMAGE_SIZE + j*OUT_IMAGE_SIZE + k] = 0;
            }
        }
    }
  
    printf("IMAGE_SIZE = %d\n", IMAGE_SIZE);
    printf("OUT_IMAGE_SIZE = %d\n", OUT_IMAGE_SIZE);
    printf("IMAGE_SIZE = %d\n", IMAGE_SIZE);
    printf("LENGTH_IN_TILE = %d\n", LENGTH_IN_TILE);
    printf("LENGTH_OUT_TILE = %d\n", LENGTH_OUT_TILE);

    // RUN CPU
    #ifdef CPU
    auto start2 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < OUT_IMAGE_SIZE; i ++) {
        printf("Host compute plane %d\n", i);
        float * data_in_plane = (float *)malloc(sizeof(float) * LENGTH_IN_TILE);
        float * data_out_plane = (float *)malloc(sizeof(float) * LENGTH_OUT_TILE);
        for(int j = 0; j < LENGTH_OUT_TILE; j++){
            data_out_plane[j] = 0;
        }
        data_in_plane = data_in + i * IMAGE_SIZE * IMAGE_SIZE;
        conv_3d_cpu_one_plane(data_in_plane, filter, data_out_plane);
        memcpy(data_out_cpu + i*OUT_IMAGE_SIZE*OUT_IMAGE_SIZE, data_out_plane, LENGTH_OUT_TILE*sizeof(float));
        //if(i == 0) {
        //    printf("index = %d, data_out_cpu[%d] = %f\n", i, 0, data_out_plane[0]);
        //}
    }
    auto diff2 = std::chrono::high_resolution_clock::now() - start2;
    auto t2 = std::chrono::duration_cast<std::chrono::nanoseconds>(diff2);
    cout << "CPU time: " << t2.count() << std::endl;
    #endif

    // RUN FPGA
    auto start1 = std::chrono::high_resolution_clock::now();
    #ifdef MCC_ACC
    __merlin_conv_3d_kernel(data_in, filter, data_out_merlin);
    #endif
    auto diff1 = std::chrono::high_resolution_clock::now() - start1;
    auto t1 = std::chrono::duration_cast<std::chrono::nanoseconds>(diff1);
    cout << "FPGA time: " << t1.count() << std::endl;

    #ifdef MCC_ACC
    __merlin_release();
    #endif

    // diff
    #ifdef CPU
    bool failed = false;
    for(int i = 0; i < DATA_OUT_LENGTH; i++){
        float diff = data_out_cpu[i] - data_out_merlin[i];
        bool pass = false;
        if (diff < 1.0 && diff > -1.0) {
            pass = true;
        }
        //if(data_out_cpu[i] != data_out_merlin[i]) {
        if(! pass) {
            failed = true;
            printf("%4d fpga :%15.6f, cpu 2 :%15.6f\n", i, data_out_merlin[i], data_out_cpu[i]);
        }
    }

    if (failed) {
        printf("Test Failed!\n");
    } else {
        printf("Test Passed!\n");
    }
    #endif
    return 0;
} 
