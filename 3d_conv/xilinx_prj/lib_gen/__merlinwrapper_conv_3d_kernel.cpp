#include "cmost.h"
#include <string.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <hls_stream.h>
#include "merlin_type_define.h"
#include "__merlinhead_kernel_top.h"
#include <chrono> 
#include <iostream>
void __merlinwrapper_conv_3d_kernel(float data_in[DATA_IN_LENGTH],
                                    float filter[FILTER_IN_LENGTH],
                                    float data_out[DATA_OUT_LENGTH]) {
    if (filter == 0) {
        printf("Error! Detected null pointer 'filter' for external memory.\n");
        exit(1);
    }
    if (data_in == 0) {
        printf("Error! Detected null pointer 'data_in' for external memory.\n");
        exit(1);
    }
    if (data_out == 0) {
      printf("Error! Detected null pointer 'data_out' for external memory.\n");
      exit(1);
    }
    
    #ifdef DEBUG
    printf("IN_SIZE_TILE = %d, OUT_SIZE_TILE = %d, IN_DEPTH = %d\n", IN_SIZE_TILE, OUT_SIZE_TILE, IN_DEPTH);
    printf("IN_SIZE_ONE_CALL = %d, OUT_SIZE_ONE_CALL = %d\n", IN_SIZE_ONE_CALL, OUT_SIZE_ONE_CALL);
    printf("LAST_OUT_SIZE_ONE_CALL = %d\n", LAST_OUT_SIZE_ONE_CALL);
    #endif
    float * data_in_merlin[PE];
    float * data_out_merlin[PE];
    for (int j = 0; j < PE; j++) {
        data_in_merlin[j] = (float *)malloc(IN_SIZE_TILE*sizeof(float));
        data_out_merlin[j] = (float *)malloc(OUT_SIZE_TILE*sizeof(float));
    }
    #ifdef DEBUG
    auto start0 = std::chrono::high_resolution_clock::now();
    #endif
    //for(int i=0; i<IMAGE_SIZE; i++) {
    for(int i=0; i<FILTER_SIZE-1; i++) {
        for(int j=0; j<IN_DEPTH; j++) {
            for(int k=0; k<IMAGE_SIZE; k++) {
                data_in_merlin[0][i*IMAGE_SIZE*IN_DEPTH + j*IMAGE_SIZE + k] = data_in[i*IMAGE_SIZE*IMAGE_SIZE + (j+STEP*0)*IMAGE_SIZE + k];
                data_in_merlin[1][i*IMAGE_SIZE*IN_DEPTH + j*IMAGE_SIZE + k] = data_in[i*IMAGE_SIZE*IMAGE_SIZE + (j+STEP*1)*IMAGE_SIZE + k];
                data_in_merlin[2][i*IMAGE_SIZE*IN_DEPTH + j*IMAGE_SIZE + k] = data_in[i*IMAGE_SIZE*IMAGE_SIZE + (j+STEP*2)*IMAGE_SIZE + k];
                data_in_merlin[3][i*IMAGE_SIZE*IN_DEPTH + j*IMAGE_SIZE + k] = data_in[i*IMAGE_SIZE*IMAGE_SIZE + (j+STEP*3)*IMAGE_SIZE + k];
            }
        }
    }
    #ifdef DEBUG
    auto diff0 = std::chrono::high_resolution_clock::now() - start0;
    auto t0 = std::chrono::duration_cast<std::chrono::nanoseconds>(diff0);
    std::cout << "Data prepare time: " << t0.count() << std::endl;
    #endif

    // filter only need to copy once
    #ifdef DEBUG
    printf("[Merlin Info] Start %s data for %s, data size = %d...\n", "copy in", "filter", FILTER_IN_LENGTH * sizeof(float ));
    fflush(stdout);
    #endif
    for (int j = 0; j < PE; j++) {
        m_q[j]->enqueueWriteBuffer(*buffer_filter[j], CL_TRUE, 
                              0, 
                              FILTER_IN_LENGTH * sizeof(float), 
                              filter);
//        memcpy(filter_align[j].data(), filter, FILTER_IN_LENGTH*sizeof(float));
//        m_q[j]->enqueueMigrateMemObjects({*(buffer_filter[j])}, 0);
    }
    
    int flag = 0;
    for (int i = 0; i < OUT_IMAGE_SIZE + OVERLAP; i++) {
//    for (int i = 0; i < 2 + OVERLAP; i++) {
        for (int j = 0; j < PE; j++) {
            int queue_index = flag % (OVERLAP*PE);
            int overlap_index = (flag/PE) % OVERLAP;
            int pe_index = j;
            //printf("queue_index = %d, overlap_index = %d, pe_index = %d\n", queue_index, overlap_index, pe_index);
            if (i > 1) {
                #ifdef DEBUG
                //printf("Wait finish for queue %d\n", queue_index);
                #endif
                m_q[queue_index]->finish();
                //printf("memcpy out index = %d, offset = %d, size = %d\n", pe_index, (i-OVERLAP)*OUT_SIZE_TILE, OUT_SIZE_ONE_CALL);
                //memcpy(data_out_merlin[pe_index] + (i-OVERLAP)*OUT_SIZE_ONE_CALL, output_align[pe_index][overlap_index].data(), OUT_SIZE_ONE_CALL*sizeof(float));
                if (j == PE - 1 ) {
                    memcpy(data_out + (i-OVERLAP)*OUT_IMAGE_SIZE*OUT_IMAGE_SIZE+j*OUT_SIZE_ONE_CALL, 
                           output_align[pe_index][overlap_index].data(), 
                           LAST_OUT_SIZE_ONE_CALL*sizeof(float));
                } else {
                    memcpy(data_out + (i-OVERLAP)*OUT_IMAGE_SIZE*OUT_IMAGE_SIZE+j*OUT_SIZE_ONE_CALL, 
                           output_align[pe_index][overlap_index].data(), 
                           OUT_SIZE_ONE_CALL*sizeof(float));
                }
//                memcpy(data_out + (i-OVERLAP)*OUT_IMAGE_SIZE*OUT_IMAGE_SIZE+j*OUT_SIZE_ONE_CALL, 
//                       output_align[pe_index][overlap_index].data(), 
//                       OUT_SIZE_ONE_CALL*sizeof(float));
                #ifdef DEBUG
                //printf("index = %d, data_out_kernel host[%d] = %15.6f\n", i, 0, (output_align[pe_index][overlap_index].data())[0]);
                #endif
            }
            if (i < OUT_IMAGE_SIZE) {
                #ifdef DEBUG
                printf("Do compute for one plane %d, on queue %d\n", i, queue_index);
                #endif
                //printf("memcpy in index = %d, offset = %d, size = %d\n", pe_index, i*IN_DEPTH*IMAGE_SIZE, IN_SIZE_ONE_CALL);
                //if(i == 0 && j == 0) {
                //    for(int k=0; k<IN_SIZE_ONE_CALL; k++) {
                //        printf("org k = %d, data = %f\n", k, data_in_merlin[pe_index][k]);
                //    }
                //}
                //if(i == 0 && j == 0) {
                //    printf("offset1 = %d, offset2 = %d\n", 
                //           (i+FILTER_SIZE-1)*IN_DEPTH*IMAGE_SIZE, 
                //           (i+FILTER_SIZE-1)*IMAGE_SIZE*IMAGE_SIZE + STEP*j*IMAGE_SIZE);
                //}
                memcpy(data_in_merlin[pe_index] + (i+FILTER_SIZE-1)*IN_DEPTH*IMAGE_SIZE, 
                       data_in + (i+FILTER_SIZE-1)*IMAGE_SIZE*IMAGE_SIZE + STEP*j*IMAGE_SIZE, 
                       IN_DEPTH*IMAGE_SIZE*sizeof(float));
                //if(i == 0 && j == 0) {
                //    for(int k=0; k<IN_SIZE_ONE_CALL; k++) {
                //        printf("new k = %d, data = %f\n", k, data_in_merlin[pe_index][k]);
                //    }
                //}
                memcpy(input_align[pe_index][overlap_index].data(), 
                       data_in_merlin[pe_index] + i*IN_DEPTH*IMAGE_SIZE, 
                       IN_SIZE_ONE_CALL*sizeof(float));
                
                //if(i == 0 && j == 0) {
                //for(int x = 0; x < IN_SIZE_ONE_CALL; x++) {
                //    printf("input_align[%d] = %f\n", x, (input_align[pe_index][overlap_index].data())[x]);
                //}
                //}
                m_q[queue_index]->enqueueMigrateMemObjects({*(buffer_input[pe_index][overlap_index])}, 0);
                conv_kernel[pe_index]->setArg(0, *(buffer_input[pe_index][overlap_index]));
                conv_kernel[pe_index]->setArg(1, *(buffer_filter[pe_index]));
                conv_kernel[pe_index]->setArg(2, *(buffer_output[pe_index][overlap_index]));
                m_q[queue_index]->enqueueTask(*conv_kernel[pe_index]); 
                m_q[queue_index]->enqueueMigrateMemObjects({*(buffer_output[pe_index][overlap_index])}, CL_MIGRATE_MEM_OBJECT_HOST);
            }
            flag++;
        }
        #ifdef DEBUG
        printf("\n");
        #endif
    }
    for (int i = 0; i < PE * OVERLAP; i++) {
        m_q[i]->finish();
    }
   /* 
    printf("Start merge out data\n");
    for(int i=0; i<OUT_IMAGE_SIZE; i++) {
        for(int j=0; j<OUT_IMAGE_SIZE; j++) {
            for(int k=0; k<OUT_IMAGE_SIZE; k++) {
                if(j < OUT_DEPTH) {
                    data_out[i*OUT_IMAGE_SIZE*OUT_IMAGE_SIZE + j*OUT_IMAGE_SIZE + k] \
                    = data_out_merlin[0][i*OUT_IMAGE_SIZE*STEP + j*OUT_IMAGE_SIZE + k];
                } else if(j < OUT_DEPTH * 2) {
                    data_out[i*OUT_IMAGE_SIZE*OUT_IMAGE_SIZE + j*OUT_IMAGE_SIZE + k] \
                    = data_out_merlin[1][i*OUT_IMAGE_SIZE*STEP + (j-OUT_DEPTH)*OUT_IMAGE_SIZE + k];
                } else if(j < OUT_DEPTH * 3) {
                    data_out[i*OUT_IMAGE_SIZE*OUT_IMAGE_SIZE + j*OUT_IMAGE_SIZE + k] \
                    = data_out_merlin[2][i*OUT_IMAGE_SIZE*STEP + (j-OUT_DEPTH*2)*OUT_IMAGE_SIZE + k];
                } else {
                    data_out[i*OUT_IMAGE_SIZE*OUT_IMAGE_SIZE + j*OUT_IMAGE_SIZE + k] \
                    = data_out_merlin[3][i*OUT_IMAGE_SIZE*STEP + (j-OUT_DEPTH*3)*OUT_IMAGE_SIZE + k];
                }
            }
        }
    }
    */
}
void __merlin_conv_3d_kernel(float data_in[LENGTH_IN_TILE],float filter[FILTER_IN_LENGTH],float data_out[LENGTH_OUT_TILE]) {
    __merlinwrapper_conv_3d_kernel(data_in,filter,data_out);
}
