#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string>
#include <fstream>

#include "BlazeTest.h"

//#define DUMP
using namespace blaze;

int main(int argc, char** argv) {

  if (argc < 3) {
#ifndef DUMP
    printf("USAGE: %s <conf_path> <n> (<num_clusters> <vector_length>)\n", argv[0]);
#else
    printf("USAGE: %s <conf_path> <input_file> \n", argv[0]);
#endif
    return -1;
  }

  try {
    BlazeTest<double, double> test(argv[1], 1e-5);

#ifndef DUMP
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

    // input
    double* data_samples = new double[data_size];
    double* centers = new double[center_size];

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
        centers[i*(feature_size+1)+j] = 0.0;
        norm += centers[i*(feature_size+1)+j]*centers[i*(feature_size+1)+j];
      }
      centers[i*(feature_size+1)+feature_size] = norm;
    }
#else    
#endif

    // setup input data for tested tasks
    test.setInput(0, data_samples, num_samples, feature_size+1);
    test.setInput(1, &num_runs, 1, 1);
    test.setInput(2, &num_clusters, 1, 1);
    test.setInput(3, centers, 1, center_size);

    // run test
    test.run();

    delete [] data_samples;
    delete [] centers;
  }
  catch (std::runtime_error &e) {
    printf("%s\n", e.what());
    return -1;
  }

  return 0;
}
