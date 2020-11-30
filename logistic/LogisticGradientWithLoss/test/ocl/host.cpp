#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <fcntl.h>
#include <string>
#include <fstream>
#include <cmath>

#define GROUP_SIZE 64
#define GROUP_NUM  1024

#ifdef GPU
#include "OpenCLEnv_gpu.h"
#else
#include "OpenCLEnv.h"
#define SUPPORT_WIDE_BUS
#define BUS_WIDTH 8
#endif

//#define DUMP

void gradient(int, int, int, double*, double*, double*);

int main(int argc, char** argv) {

  if (argc < 3) {
#ifndef DUMP
    printf("USAGE: %s <bit_path> <num_samples>\n", argv[0]);
#else
    printf("USAGE: %s <bit_path> <dump_file>\n", argv[0]);
#endif
    return -1;
  }

  struct	timeval t1, t2, tr;

  try { 
    OpenCLEnv env(argv[1], "gradient");

    cl_context       context = env.getContext();
    cl_kernel        kernel  = env.getKernel();
#ifdef GPU
    cl_command_queue cmd     = env.getCmdQueue(0); 
    cl_program       program = env.getProgram();
#else
    cl_command_queue cmd     = env.getCmdQueue(); 
#endif

    int err;

#ifndef DUMP
    // prepare data
    int feature_size = 784;
    int num_labels = 10;
    int num_samples = atoi(argv[2]);

    if (argc > 4) {
      num_labels = atoi(argv[3]);
      feature_size = atoi(argv[4]);
    }

    int data_size = (feature_size+1) * num_samples;
    int weight_size = (num_labels-1) * feature_size;

    // input
    double* data_samples = new double[data_size];
    double* weights = new double[weight_size];

    // setup input with random data
    for (int i=0; i<num_samples; i++) {
      data_samples[i*(feature_size+1)] = rand()%num_labels; 
      for (int j=0; j<feature_size; j++) {
        data_samples[i*(feature_size+1)+1+j] = (double)rand()/RAND_MAX;
      }
    }
    for (int i=0; i<weight_size; i++) {
      weights[i] = (double)rand()/RAND_MAX;
    }
#else    
    FILE* fin = fopen(argv[2], "rb");
    if (!fin) {
      throw std::runtime_error("cannot find input file");
    }
    // prepare data
    int num_samples;
    int weight_size;
    int feature_size;
    int num_labels;

    fread(&num_samples, sizeof(int), 1, fin);
    fread(&feature_size, sizeof(int), 1, fin);
    fread(&weight_size, sizeof(int), 1, fin);
    fread(&num_labels, sizeof(int), 1, fin);

    int data_size = (feature_size+1) * num_samples;

    double* data_samples = new double[data_size];
    double* weights = new double[weight_size];

    // input
    fread(data_samples, sizeof(double), data_size, fin);
    fread(weights, sizeof(double), weight_size, fin);

    fclose(fin);
#endif
    // output
    double* output_base = new double[weight_size+1];
    double* output_test = new double[weight_size+1];

    // run test
    cl_event event;

#ifdef SUPPORT_WIDE_BUS
    int aligned_feature_size = (feature_size+BUS_WIDTH)/BUS_WIDTH*BUS_WIDTH;
    int aligned_data_size = aligned_feature_size*num_samples;
    cl_mem cl_data    = clCreateBuffer(context, CL_MEM_READ_ONLY,  aligned_data_size*sizeof(double), NULL, NULL);
#else
    cl_mem cl_data    = clCreateBuffer(context, CL_MEM_READ_ONLY,  data_size*sizeof(double), NULL, NULL);
#endif
    cl_mem cl_weights = clCreateBuffer(context, CL_MEM_READ_ONLY,  weight_size*sizeof(double), NULL, NULL);
    cl_mem cl_output  = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (weight_size+1)*sizeof(double), NULL, NULL);

#ifdef SUPPORT_WIDE_BUS
    for (int k=0; k<num_samples; k++) {
      err  = clEnqueueWriteBuffer(cmd, cl_data, CL_TRUE, 
          k*aligned_feature_size*sizeof(double), 
          (feature_size+1)*sizeof(double), 
          data_samples+k*(feature_size+1), 
          0, NULL, NULL);
    }
#else
    err  = clEnqueueWriteBuffer(cmd, cl_data, CL_TRUE, 0, data_size*sizeof(double), data_samples, 0, NULL, NULL);
#endif
    err |= clEnqueueWriteBuffer(cmd, cl_weights, CL_TRUE, 0, weight_size*sizeof(double), weights, 0, NULL, &event);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Cannot write buffers\n");
    }

#ifdef GPU
    cl_mem cl_temp = clCreateBuffer(context, CL_MEM_READ_WRITE, GROUP_NUM*(weight_size+1)*sizeof(double), NULL, NULL);

    cl_kernel sum_kernel = clCreateKernel(program, "vector_sum", &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Cannot create vector_sum kernel\n");
    }
    err  = clSetKernelArg(kernel, 0, sizeof(int), &num_samples);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &num_labels);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &feature_size);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_weights);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_data);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &cl_temp);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Cannot set kernel args\n");
    }
    size_t work_lsize[1];
    size_t work_gsize[1];

    for (int iter=0; iter<3; iter++) {

      work_lsize[0] = GROUP_SIZE;
      work_gsize[0] = GROUP_NUM*work_lsize[0];

      gettimeofday(&t1, NULL);
      err = clEnqueueNDRangeKernel(cmd, kernel, 1, NULL, 
          work_gsize, work_lsize, 0, NULL, &event);
      clWaitForEvents(1, &event);

      int vector_size = weight_size+1;
      int num_vectors = GROUP_NUM;

      err  = clSetKernelArg(sum_kernel, 0, sizeof(int), &vector_size);
      err |= clSetKernelArg(sum_kernel, 1, sizeof(int), &num_vectors);
      err |= clSetKernelArg(sum_kernel, 2, sizeof(cl_mem), &cl_temp);
      err |= clSetKernelArg(sum_kernel, 3, sizeof(cl_mem), &cl_output);
      if (err != CL_SUCCESS) {
        throw std::runtime_error("Cannot set sum_kernel args\n");
      }

      work_lsize[0] = 1024;
      work_gsize[0] = work_lsize[0];

      err = clEnqueueNDRangeKernel(cmd, sum_kernel, 1, NULL, 
          work_gsize, work_lsize, 0, NULL, &event);

      if (err != CL_SUCCESS) {
        throw std::runtime_error("Cannot enqueue sum kernel\n");
      }
      clWaitForEvents(1, &event);

      gettimeofday(&t2, NULL);
      timersub(&t1, &t2, &tr);
      fprintf(stdout, "Kernel execution takes %.4f ms\n", 
          fabs(tr.tv_sec*1000.0+(double)tr.tv_usec/1000.0));
    }

    err = clEnqueueReadBuffer(cmd, cl_output, CL_TRUE, 0, (weight_size+1)*sizeof(double), output_test, 0, NULL, &event);  
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Cannot read output buffer\n");
    }
    clWaitForEvents(1, &event);

    /*
       double* output_temp = new double[GROUP_NUM*(weight_size+1)];
       err = clEnqueueReadBuffer(cmd, cl_temp, CL_TRUE, 0, GROUP_NUM*(weight_size+1)*sizeof(double), output_temp, 0, NULL, &event);  
       if (err != CL_SUCCESS) {
       throw std::runtime_error("Cannot read output buffer\n");
       }
       clWaitForEvents(1, &event);

       for (int j=0; j<weight_size+1; j++) {
       for (int i=0; i<GROUP_NUM; i++) {
       if (i==0) {
       output_test[j] = output_temp[i*(weight_size+1)+j];
       }    
       else {
       output_test[j] += output_temp[i*(weight_size+1)+j];
       }
       }
       }

       delete [] output_temp;
       */
#else
    err  = clSetKernelArg(kernel, 0, sizeof(int), &num_samples);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &num_labels);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &feature_size);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_weights);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_data);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &cl_output);

    gettimeofday(&t1, NULL);
    err = clEnqueueTask(cmd, kernel, 0, NULL, &event);
    clWaitForEvents(1, &event);

    gettimeofday(&t2, NULL);
    timersub(&t1, &t2, &tr);
    fprintf(stdout, "Kernel execution takes %.4f ms\n", 
        fabs(tr.tv_sec*1000.0+(double)tr.tv_usec/1000.0));

    err = clEnqueueReadBuffer(cmd, cl_output, CL_TRUE, 0, (weight_size+1)*sizeof(double), output_test, 0, NULL, &event);  
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Cannot read output buffer\n");
    }

    clWaitForEvents(1, &event);

#endif

    // compute baseline results
    gradient(num_samples, num_labels, feature_size, weights, data_samples, output_base);

    // compare results
    double diff_total = 0.0;
    double diff_ratio = 0.0;
    double max_diff = 0.0;
    for (int k=0; k<weight_size+1; k++) {
      double diff = std::abs(output_base[k] - output_test[k]); 
      if (diff > max_diff) {
        max_diff = diff;
      }

      diff_total += diff;
      if (output_base[k]!=0) {
        diff_ratio += diff / std::abs(output_base[k]);
      }

      if (k<10) {
      //if (diff / std::abs(output_base[k]) > 0.05) {
        printf("%d: %f|%f, ratio=%f\n", 
            k,
            output_base[k], 
            output_test[k],
            diff/std::abs(output_base[k]));
      }
    }
    printf("diff: %f max, %f per point, %f%%/per point\n",
        max_diff,
        diff_total/(weight_size+1),
        diff_ratio/(weight_size+1)*100.0);

    clReleaseMemObject(cl_weights);
    clReleaseMemObject(cl_data);
    clReleaseMemObject(cl_output);

    delete output_base;
    delete output_test;
    delete data_samples;
    delete weights;
  } 
  catch (std::runtime_error &e) {
    printf("%s\n", e.what());
    return -1;
  }

  return 0;
}
