#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <fcntl.h>
#include <string>
#include <fstream>
#include <cmath>

#include "OpenCLEnv.h"
#include "baseline.h"

//#define DUMP
#define SUPPORT_WIDE_BUS
#define BUS_WIDTH 8

int main(int argc, char** argv) {

  if (argc < 3) {
    printf("USAGE: %s <bit_path> <num_samples>\n", argv[0]);
    return -1;
  }

  struct	timeval t1, t2, tr;

  try { 
    OpenCLEnv env(argv[1], "kmeans");

    cl_context       context = env.getContext();
    cl_kernel        kernel  = env.getKernel();
    cl_command_queue cmd     = env.getCmdQueue(); 

    int err;

    // prepare data
    int feature_size = 784;
    int num_runs = 1;         // for test set num_runs to be 1
    int num_clusters = 10;
    int num_samples = atoi(argv[2]);

    if (argc > 4) {
      num_clusters = atoi(argv[3]);
      feature_size = atoi(argv[4]);
    }

    int data_size = (feature_size+1) * num_samples;
    int center_size = num_clusters * num_runs * (feature_size+1);
    int output_size = num_clusters * num_runs * (feature_size+3);

    double* data_samples = new double[data_size];
    double* centers = new double[center_size];
    double* output_base = new double[output_size];
    double* output_test = new double[output_size];

    // setup input with random data
    for (int i=0; i<num_samples; i++) {
      double norm = 0.0;
      for (int j=0; j<feature_size; j++) {
        data_samples[i*(feature_size+1)+j] = (double)rand()/RAND_MAX;
        norm += data_samples[i*(feature_size+1)+j]*data_samples[i*(feature_size+1)+j];
      }
      norm = sqrt(norm);
      data_samples[i*(feature_size+1)+feature_size] = norm;
    }
    for (int i=0; i<num_clusters*num_runs; i++) {
      double norm = 0.0;
      for (int j=0; j<feature_size; j++) {
        centers[i*(feature_size+1)+j] = (double)rand()/RAND_MAX;
        norm += centers[i*(feature_size+1)+j]*centers[i*(feature_size+1)+j];
      }
      centers[i*(feature_size+1)+feature_size] = norm;
    }
    // run test
    cl_event event;

#ifdef SUPPORT_WIDE_BUS
    int feature_size_u512 = (feature_size+BUS_WIDTH)/BUS_WIDTH*BUS_WIDTH;
    int aligned_data_size = feature_size_u512*num_samples;

    cl_mem cl_data    = clCreateBuffer(context, CL_MEM_READ_ONLY,  aligned_data_size*sizeof(double), NULL, NULL);
#else
    cl_mem cl_data    = clCreateBuffer(context, CL_MEM_READ_ONLY,  data_size*sizeof(double), NULL, NULL);
#endif
    cl_mem cl_centers = clCreateBuffer(context, CL_MEM_READ_ONLY,  center_size*sizeof(double), NULL, NULL);
    cl_mem cl_output  = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size*sizeof(double), NULL, NULL);

#ifdef SUPPORT_WIDE_BUS
    for (int k=0; k<num_samples; k++) {
      err  = clEnqueueWriteBuffer(cmd, cl_data, CL_TRUE, 
          k*feature_size_u512*sizeof(double), 
          (feature_size+1)*sizeof(double), 
          data_samples+k*(feature_size+1), 
          0, NULL, NULL);
    }
#else
    err  = clEnqueueWriteBuffer(cmd, cl_data, CL_TRUE, 0, data_size*sizeof(double), data_samples, 0, NULL, NULL);
#endif
    err |= clEnqueueWriteBuffer(cmd, cl_centers, CL_TRUE, 0, center_size*sizeof(double), centers, 0, NULL, &event);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Cannot write buffers\n");
    }
    clWaitForEvents(1, &event);

    err  = clSetKernelArg(kernel, 0, sizeof(int), &num_samples);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &num_runs);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &num_clusters);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &feature_size);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_data);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &cl_centers);
    err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &cl_output);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Cannot set kernel args\n");
    }

    for (int iter=0; iter<3; iter++) {
	gettimeofday(&t1, NULL);

	err = clEnqueueTask(cmd, kernel, 0, NULL, &event);
	clWaitForEvents(1, &event);

	gettimeofday(&t2, NULL);
	timersub(&t1, &t2, &tr);
	fprintf(stdout, "FPGA execution takes %.4f ms\n", 
		fabs(tr.tv_sec*1000.0+(double)tr.tv_usec/1000.0));
    }

    err = clEnqueueReadBuffer(cmd, cl_output, CL_TRUE, 0, 
			      output_size*sizeof(double), output_test, 0, NULL, &event);  

    if (err != CL_SUCCESS) {
	throw std::runtime_error("Cannot read output buffer\n");
    }
    clWaitForEvents(1, &event);

    // compute baseline results
    kmeans_base(num_samples, num_runs, num_clusters, feature_size, data_samples, centers, output_base);

    // compare results
    double diff_total = 0.0;
    double diff_ratio = 0.0;
    double max_diff = 0.0;
    //printf("output_size=%d\n", output_size);
    for (int k=0; k<output_size; k++) {
	/*    printf("%d: %f|%f\n", 
            k,
            output_base[k], 
            output_test[k]);
	*/
      double diff = std::abs(output_base[k] - output_test[k]); 
      if (diff > max_diff) {
        max_diff = diff;
      }

      diff_total += diff;
      if (output_base[k]!=0) {
        diff_ratio += diff / std::abs(output_base[k]);
      }
      if (diff / std::abs(output_base[k]) > 1e-4) {
        /*
        printf("%d: %f|%f, ratio=%f\n", 
            k,
            output_base[k], 
            output_test[k],
            diff/std::abs(output_base[k]));
            */
      }
    }
    printf("diff: %f max, %f/point, %f%/point\n",
        max_diff,
        diff_total/(output_size+1),
        diff_ratio/(output_size+1)*100.0);

    clReleaseMemObject(cl_centers);
    clReleaseMemObject(cl_data);
    clReleaseMemObject(cl_output);

    delete [] output_base;
    delete [] output_test;
    delete [] data_samples;
    delete [] centers;
  } 
  catch (std::runtime_error &e) {
    printf("%s\n", e.what());
    return -1;
  }

  return 0;
}
