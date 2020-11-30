#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <limits>

#include "blaze.h" 

using namespace blaze;

void kmeans(int, int, int, int, double*, double*, double*);

class KMeansContrib_ctest : public Task {
public:

  // extends the base class constructor
  // to indicate how many input blocks
  // are required
  KMeansContrib_ctest(): Task(4) {;}

  // overwrites the compute function
  // Input data:
  // - data: layout as num_samples x [double[] vector, double norm]
  // - num_runs
  // - num_clusters
  // - centers: layout as runs x k x [double[] vector, double norm]
  // Output data:
  // - tuple of sum and counts: 
  //   [i, j, double[] sum(i)(j), double count(i)(j)]
  virtual void compute() {

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

    // get the pointer to input/output data
    double * data     = (double*)getInput(0);
    double * centers  = (double*)getInput(3);
    double * output   = (double*)getOutput(0, 
                         data_length+2, num_runs*num_clusters, 
                         sizeof(double));

    if (!data || !centers || !output) {
      throw std::runtime_error("Cannot get data pointers");
    }

    kmeans(num_samples, num_runs, num_clusters, vector_length,
        data, centers, output);
  }
};

extern "C" Task* create() {
  return new KMeansContrib_ctest();
}

extern "C" void destroy(Task* p) {
  delete p;
}
