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
    printf("USAGE: %s <conf_path> <n> (<label_size> <feature_size>)\n", argv[0]);
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

    // setup input data for tested tasks
    test.setInput(0, data_samples, num_samples, feature_size+1);
    test.setInput(1, weights, 1, weight_size);

    // run test
    test.run();

    delete data_samples;
    delete weights;
  }
  catch (std::runtime_error &e) {
    printf("%s\n", e.what());
    return -1;
  }

  return 0;
}
