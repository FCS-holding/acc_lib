#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <fcntl.h>
#include <string>
#include <fstream>
#include <cmath>

#include "baseline.h"

#ifndef HLS_SIM
#define SUPPORT_WIDE_BUS
#endif
#ifndef SUPPORT_WIDE_BUS
void kmeans(int, int, int, int, double*, double*, double*);
#else
#include "ap_int.h"
#define BUS_WIDTH 8
void kmeans(int, int, int, int, ap_uint<512>*, double*, double*);
#endif

int main(int argc, char** argv) {

  if (argc < 2) {
    printf("USAGE: %s <num_samples> (<num_clusters> <feature_size>)\n", argv[0]);
    return -1;
  }
  struct	timeval t1, t2, tr;

  // prepare data
  int feature_size = 784;
  int num_runs = 1;         // for test set num_runs to be 1
  int num_clusters = 10;
  int num_samples = atoi(argv[1]);

  if (argc > 3) {
    num_clusters = atoi(argv[2]);
    feature_size = atoi(argv[3]);
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
  kmeans_base(num_samples, num_runs, num_clusters, feature_size, data_samples, centers, output_base);

  // compute test results
#ifdef SUPPORT_WIDE_BUS
  kmeans(num_samples, num_runs, num_clusters, feature_size, bus_data, centers, output_test);
#else
  kmeans(num_samples, num_runs, num_clusters, feature_size, data_samples, centers, output_test);
#endif

  // compare results
  double diff_total = 0.0;
  double diff_ratio = 0.0;
  double max_diff = 0.0;
  for (int k=0; k<output_size; k++) {
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
      diff_total/(output_size),
      diff_ratio/(output_size)*100.0);

  delete [] output_base;
  delete [] output_test;
  delete [] data_samples;
  delete [] centers;

  return 0;
}
