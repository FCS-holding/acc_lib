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

using namespace blaze;

class KMeans : public Task {
public:

  // extends the base class constructor
  // to indicate how many input blocks
  // are required
  KMeans(): Task(4) {
    addConfig(0, "align_width", "64");
  }

  // overwrites the compute function
  // Input data:
  // - data: layout as num_samples x [double norm, double[] feature]
  // - weight: (num_labels-1) x feature_length
  // Output data:
  // - gradient plus loss: [double[] gradient, double loss]
  virtual void compute() {

    struct	timeval t1, t2, tr;

    try {
      // dynamically cast the TaskEnv to OpenCLEnv
      OpenCLEnv* ocl_env = (OpenCLEnv*)getEnv();

      // get input data length
      int num_samples   = getInputNumItems(0);
      int data_length   = getInputLength(0)/num_samples;
      int num_runs      = *(reinterpret_cast<int*>(getInput(1)));
      int num_clusters  = *(reinterpret_cast<int*>(getInput(2)));
      int center_length = getInputLength(3);
      int vector_length = data_length - 1;

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
      cl_kernel        kernel  = ocl_env->getKernel();
      cl_command_queue command = ocl_env->getCmdQueue();

      int err;
      cl_event event;

      // get the pointer to input/output data
      cl_mem data     = *(cl_mem*)getInput(0);
      cl_mem centers  = *(cl_mem*)getInput(3);
      cl_mem output   = *(cl_mem*)getOutput(0, 
          data_length+2, num_runs*num_clusters, 
          sizeof(double));

      if (!data || !centers || !output) {
        throw std::runtime_error("Cannot get data pointers");
      }

      // Set the arguments to our compute kernel
      err  = clSetKernelArg(kernel, 0, sizeof(int), &num_samples);
      err |= clSetKernelArg(kernel, 1, sizeof(int), &num_runs);
      err |= clSetKernelArg(kernel, 2, sizeof(int), &num_clusters);
      err |= clSetKernelArg(kernel, 3, sizeof(int), &vector_length);
      err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &data);
      err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &centers);
      err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &output);

      if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set gradients!");
      }

      //gettimeofday(&t1, NULL);

      // Execute the kernel over the entire range of our 1d input data set
      // using the maximum number of work group items for this device

      ocl_env->lock();

      err = clEnqueueTask(command, kernel, 0, NULL, &event);

      ocl_env->unlock();
      clWaitForEvents(1, &event);

      if (err) {
        throw("Failed to execute kernel!");
      }

      /*
      err = clEnqueueReadBuffer(command, 
          gradients, CL_TRUE, 0, sizeof(float) * 10, 
          gradient_scope, 0, NULL, &event );  

      clWaitForEvents(1, &event);

      for (int i=0; i<10; i++) {
        printf("%f, %f\n", weight_scope[i], gradient_scope[i]);
      }
      */

      //gettimeofday(&t2, NULL);
      //timersub(&t1, &t2, &tr);
      //fprintf(stdout, "FPGA execution takes %.4f ms\n", 
      //    fabs(tr.tv_sec*1000.0+(double)tr.tv_usec/1000.0));
    }
    catch (std::runtime_error &e) {
      throw e;
    }
  }
};

extern "C" Task* create() {
  return new KMeans();
}

extern "C" void destroy(Task* p) {
  delete p;
}
