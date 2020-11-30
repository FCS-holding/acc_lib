#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>

#include "Task.h" 
#include "OpenCLEnv.h" 

#define GROUP_SIZE 128
#define GROUP_NUM  256

using namespace blaze;

class KMeans : public Task {
public:

  // extends the base class constructor
  // to indicate how many input blocks
  // are required
  KMeans(): Task(4) {;}

  // overwrites the compute function
  // Input data:
  // - data: layout as num_samples x [double label, double[] feature]
  // - weight: (num_labels-1) x feature_length
  // Output data:
  // - gradient plus loss: [double[] gradient, double loss]
  virtual void compute() {

    struct	timeval t1, t2, tr;

    // dynamically cast the TaskEnv to OpenCLTaskEnv
    OpenCLTaskEnv* ocl_env = (OpenCLTaskEnv*)getEnv();

    // get input data length
    int num_samples   = getInputNumItems(0);
    int data_length   = getInputLength(0)/num_samples;
    int num_runs      = *(reinterpret_cast<int*>(getInput(1)));
    int num_clusters  = *(reinterpret_cast<int*>(getInput(2)));
    int center_length = getInputLength(3);
    int vector_length = data_length - 1;
    int output_size   = num_clusters * num_runs * (data_length+2);

    // check input size
    if (center_length != num_runs*num_clusters*data_length || 
        num_runs < 1)
    {
      fprintf(stderr, "runs=%d, k=%d, num_samples=%d, data_length=%d\n", 
          num_runs, num_clusters, num_samples, data_length);
      throw std::runtime_error("Invalid input data dimensions");
    }

    // get OpenCL context
    cl_context       context = ocl_env->getContext();
    cl_command_queue command = ocl_env->getCmdQueue();
    cl_program       program = ocl_env->getProgram();

    int err;
    cl_event event;

    // get corresponding cl_kernel
    cl_kernel kernel_compute = clCreateKernel(program, "kmeans", &err);
    if (!kernel_compute || err != CL_SUCCESS) {
      throw std::runtime_error(
          "Failed to create compute kernel gradient");
    }

    cl_kernel kernel_sum = clCreateKernel(program, "vector_sum", &err);
    if (!kernel_sum || err != CL_SUCCESS) {
      throw std::runtime_error(
          "Failed to create compute kernel vector_sum");
    }

    // get the pointer to input/output data
    cl_mem data     = *(cl_mem*)getInput(0);
    cl_mem centers  = *(cl_mem*)getInput(3);
    cl_mem output   = *(cl_mem*)getOutput(0, 
        data_length+2, num_runs*num_clusters, 
        sizeof(double));

    if (!data || !centers || !output) {
      throw std::runtime_error("Cannot get data pointers");
    }

    cl_mem temp = clCreateBuffer(context, CL_MEM_READ_WRITE, 
        GROUP_NUM*output_size*sizeof(double), NULL, &err);

    if (!temp || err != CL_SUCCESS) {
      throw std::runtime_error("Cannot create ocl_temp buffer");
    }

    // Set the arguments of gradient kernel
    err  = clSetKernelArg(kernel_compute, 0, sizeof(int), &num_samples);
    err |= clSetKernelArg(kernel_compute, 1, sizeof(int), &num_runs);
    err |= clSetKernelArg(kernel_compute, 2, sizeof(int), &num_clusters);
    err |= clSetKernelArg(kernel_compute, 3, sizeof(int), &vector_length);
    err |= clSetKernelArg(kernel_compute, 4, sizeof(cl_mem), &data);
    err |= clSetKernelArg(kernel_compute, 5, sizeof(cl_mem), &centers);
    err |= clSetKernelArg(kernel_compute, 6, sizeof(cl_mem), &temp);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Cannot set gradient args");
    }

    // Set the arguments of vector_sum kernel
    int num_vectors = GROUP_NUM;

    err  = clSetKernelArg(kernel_sum, 0, sizeof(int), &num_runs);
    err |= clSetKernelArg(kernel_sum, 1, sizeof(int), &num_clusters);
    err |= clSetKernelArg(kernel_sum, 2, sizeof(int), &vector_length);
    err |= clSetKernelArg(kernel_sum, 3, sizeof(int), &num_vectors);
    err |= clSetKernelArg(kernel_sum, 4, sizeof(cl_mem), &temp);
    err |= clSetKernelArg(kernel_sum, 5, sizeof(cl_mem), &output);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Cannot set vector_sum args");
    }

    // setup workgroup dimensions
    size_t work_lsize[1];
    size_t work_gsize[1];

    work_lsize[0] = GROUP_SIZE;
    work_gsize[0] = GROUP_NUM*work_lsize[0];

    err = clEnqueueNDRangeKernel(command, kernel_compute, 1, NULL, 
        work_gsize, work_lsize, 0, NULL, &event);

    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to enqueue kmeans compute kernel");
    }
    clWaitForEvents(1, &event);

    work_lsize[0] = 1024;
    work_gsize[0] = work_lsize[0];

    err = clEnqueueNDRangeKernel(command, kernel_sum, 1, NULL, 
        work_gsize, work_lsize, 0, NULL, &event);

    if (err) {
      throw std::runtime_error("Failed to enqueue vector_sum kernel");
    }
    clWaitForEvents(1, &event);

    clReleaseMemObject(temp);
    clReleaseKernel(kernel_compute);
    clReleaseKernel(kernel_sum);
  }
};

extern "C" Task* create() {
  return new KMeans();
}

extern "C" void destroy(Task* p) {
  delete p;
}
