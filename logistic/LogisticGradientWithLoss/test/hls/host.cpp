#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <fcntl.h>
#include <string>
#include <fstream>
#include <cmath>

#include "baseline.h"

#define BUS_WIDTH 8

#ifndef SUPPORT_WIDE_BUS
void gradient(int, int, int, double*, double*, double*);
#else
#include "ap_int.h"
void gradient(int, int, int, double*, ap_uint<512>*, double*);
#endif

int main(int argc, char** argv) {

  if (argc < 2) {
    printf("USAGE: %s <num_samples> (<label_size> <feature_size>)\n", argv[0]);
    return -1;
  }
  struct	timeval t1, t2, tr;

  // prepare data
  int feature_size = 784;
  int num_labels = 10;
  int num_samples = atoi(argv[1]);

  if (argc > 3) {
    num_labels = atoi(argv[2]);
    feature_size = atoi(argv[3]);
  }

  int weight_size = (num_labels-1) * feature_size;
  int data_size = (feature_size+1) * num_samples;

  double* data_samples = new double[data_size];
  double* weights = new double[weight_size];
  double* output_base = new double[weight_size+1];
  double* output_test = new double[weight_size+1];

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

#ifdef SUPPORT_WIDE_BUS
  int feature_size_u512 = (feature_size+BUS_WIDTH)/BUS_WIDTH*BUS_WIDTH;
  ap_uint<512>* bus_data = new ap_uint<512>[num_samples*feature_size_u512/BUS_WIDTH];

  for (int i=0; i<num_samples; i++) {
    for (int j=0; j<feature_size_u512/BUS_WIDTH; j++) {
      ap_uint<512> tmp = 0;
      for (int k=0; k<BUS_WIDTH; k++) {
        if (j*BUS_WIDTH+k < feature_size+1) {
          tmp.range((k+1)*64-1, k*64) = *(long long*)(&data_samples[i*(feature_size+1)+j*BUS_WIDTH+k]);
        }
      }
      bus_data[i*feature_size_u512/BUS_WIDTH+j] = tmp;
    }
  }
#endif

  // compute baseline results
  gradient_base(num_samples, num_labels, feature_size, weights, data_samples, output_base);

  // compute test results
#ifdef SUPPORT_WIDE_BUS
  gradient(num_samples, num_labels, feature_size, weights, bus_data, output_test);
#else
  gradient(num_samples, num_labels, feature_size, weights, data_samples, output_test);
#endif

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

    if (diff / std::abs(output_base[k]) > 1e-4) {
      printf("%d: %f|%f, ratio=%f\n", 
          k,
          output_base[k], 
          output_test[k],
          diff/std::abs(output_base[k]));
    }
  }
  printf("diff: %f max, %f/point, %f%/point\n",
      max_diff,
      diff_total/(weight_size+1),
      diff_ratio/(weight_size+1)*100.0);

  delete [] output_base;
  delete [] output_test;
  delete [] data_samples;
  delete [] weights;

  return 0;
}
