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
#include "baseline.h"

using namespace blaze;

class KMeansContrib : public Task {
public:

  // extends the base class constructor
  // to indicate how many input blocks
  // are required
  KMeansContrib(): Task(4) {;}

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

    // perform computation
    double* sums = new double[num_runs*num_clusters*vector_length];
    int*    counts = new int[num_runs*num_clusters];

    memset(sums, 0, sizeof(double)*(vector_length*num_runs*num_clusters));
    memset(counts, 0, sizeof(int)*(num_runs*num_clusters));

    kmeans_base(num_samples, num_runs, num_clusters, vector_length,
        data, centers, output);
  }

private:

  inline void axpy(double alpha, double* v1, double* v2, int n) {
    for (int k=0; k<n; k++) {
      v2[k] += alpha*v1[k];
    }
  }

  inline double dot(double* v1, double* v2, int n) {
    double res = 0.0;
    for (int k=0; k<n; k++) {
      res += v1[k]*v2[k];
    }
    return res;
  }

  inline double dist(double* v1, double* v2, int n) {
    double res = 0.0;
    for (int k=0; k<n; k++) {
      res += (v1[k]-v2[k])*(v1[k]-v2[k]);
    }
    return res;
  }

  inline double fastSquareDistance(
      double* v1, double norm1,
      double* v2, double norm2,
      int n, 
      double precision = 1e-6) 
  {
    double sumSquaredNorm = norm1 * norm1 + norm2 * norm2;
    double normDiff = norm1 - norm2;
    double sqDist = 0.0;

    double precisionBound1 = 2.0 * DBL_EPSILON * sumSquaredNorm / 
                             (normDiff * normDiff + DBL_EPSILON);
    
    if (precisionBound1 < precision) {
      sqDist = sumSquaredNorm - 2.0 * dot(v1, v2, n);
    } 
    // skip Sparse vector case
    else {
      sqDist = dist(v1, v2, n);
    }
    return sqDist;
  }
};

extern "C" Task* create() {
  return new KMeansContrib();
}

extern "C" void destroy(Task* p) {
  delete p;
}
