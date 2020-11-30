#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <limits>

//#define USEMKL

#ifdef USEMKL
#include <mkl.h>
#endif

#include "blaze.h" 

using namespace blaze;

void gradient(int, int, int, double*, double*, double*);

class LogisticGradientWithLossTest : public Task {
public:

  // extends the base class constructor
  // to indicate how many input blocks
  // are required
  LogisticGradientWithLossTest(): Task(2) {;}

  // overwrites the compute function
  // Input data:
  // - data: layout as num_samples x [double label, double[] feature]
  // - weight: (num_labels-1) x feature_length
  // Output data:
  // - gradient plus loss: [double[] gradient, double loss]
  virtual void compute() {

    // get input data length
    int data_length = getInputLength(0);
    int num_samples = getInputNumItems(0);
    int weight_length = getInputLength(1);
    int feature_length = data_length / num_samples - 1;
    int num_labels = weight_length / feature_length + 1;

    // check input size
    if (weight_length % feature_length != 0 || 
        num_labels < 2)
    {
      fprintf(stderr, "num_samples=%d, feature_length=%d, weight_length=%d\n", num_samples, feature_length, weight_length);
      throw std::runtime_error("Invalid input data dimensions");
    }

    // get the pointer to input/output data
    double * data     = (double*)getInput(0);
    double * weights  = (double*)getInput(1);
    double * output   = (double*)getOutput(0, weight_length+1, 1, sizeof(double));

    if (!data || !weights || !output) {
      throw std::runtime_error("Cannot get data pointers");
    }

    // perform computation
    int L = num_labels;
    int D = feature_length;

    memset(output, 0, sizeof(double)*(weight_length+1));

    gradient(num_samples, L, D, weights, data, output);

  }
};

extern "C" Task* create() {
  return new LogisticGradientWithLossTest();
}

extern "C" void destroy(Task* p) {
  delete p;
}
