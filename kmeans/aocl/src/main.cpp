#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <fcntl.h>
#include <string>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <fstream>

using namespace std;
#include "baseline.h"
#include "kmeans.h"
#ifdef MCC_ACC
  #include "__merlinkmeans_kernel.h"
#endif
#define AOCL_ALIGNMENT 64
#define N_ITER 1 
//#define DUMP
//#define SUPPORT_WIDE_BUS
//#define BUS_WIDTH 8
//
void *alignedMalloc(size_t size) {
  void *result = NULL;
  posix_memalign (&result, AOCL_ALIGNMENT, size);
  return result;
}

int main(int argc, char** argv) {

  if (argc < 3) {
    printf("USAGE: %s <bit_path> <num_samples>\n", argv[0]);
    return -1;
  }

  struct	timeval t1, t2, tr;

  try { 
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
/*
    double* data_samples = new double[data_size];
    double* centers = new double[center_size];
    double* output_base = new double[output_size];
    double* output_test = new double[output_size];
*/

    //int adata_size=sizeof(double)*((data_size/AOCL_ALIGNMENT) + 1)*AOCL_ALIGNMENT;
    //int acenter_size=sizeof(double)*((center_size/AOCL_ALIGNMENT) + 1)*AOCL_ALIGNMENT;
    //int aoutput_size=sizeof(double)*((output_size/AOCL_ALIGNMENT) + 1)*AOCL_ALIGNMENT;
    int adata_size=sizeof(double)*((((D+1)*MAX_NUM_SAMPLES)/AOCL_ALIGNMENT) + 1)*AOCL_ALIGNMENT;
    int acenter_size=sizeof(double)*((((D+1)*MAX_NUM_CLUSTERS*MAX_NUM_RUNS)/AOCL_ALIGNMENT) + 1)*AOCL_ALIGNMENT;
    int aoutput_size=sizeof(double)*((((D+3)*MAX_NUM_CLUSTERS*MAX_NUM_RUNS)/AOCL_ALIGNMENT) + 1)*AOCL_ALIGNMENT;
    double* data_samples = (double*)alignedMalloc(adata_size);
    double* centers = (double*)alignedMalloc(acenter_size);
    double* output_base = (double*)alignedMalloc(aoutput_size);
    double* output_test = (double*)alignedMalloc(aoutput_size);


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

    ofstream fout;
    fout.open("../data/data_out.txt");
    for (int iter=0; iter<N_ITER; iter++) {
	    gettimeofday(&t1, NULL);

    #ifdef MCC_ACC
        __merlin_init(argv[argc - 1]);
        __merlinwrapper_kmeans_kernel(num_samples, num_runs, num_clusters, feature_size, data_samples, centers, output_test, adata_size, acenter_size, aoutput_size);
    #else
        kmeans(num_samples, num_runs, num_clusters, feature_size, data_samples, centers, output_test, adata_size, acenter_size, aoutput_size); 
        //kmeans_base(num_samples, num_runs, num_clusters, feature_size, data_samples, centers, output_test);
    #endif
	gettimeofday(&t2, NULL);
	timersub(&t1, &t2, &tr);
	fprintf(stdout, "FPGA execution takes %.4f ms\n", 
		fabs(tr.tv_sec*1000.0+(double)tr.tv_usec/1000.0));
    }
    for (int i=0; i<output_size; i++) {
      fout << output_test[i] << "\n";
    }
    fout.close();

    fout.open("../data/data_out_base.txt");
    // compute baseline results
    for (int iter=0; iter<N_ITER; iter++) {
        gettimeofday(&t1, NULL);
        kmeans_base(num_samples, num_runs, num_clusters, feature_size, data_samples, centers, output_base);
        gettimeofday(&t2, NULL);
        timersub(&t1, &t2, &tr);
        fprintf(stdout, "CPU execution takes %.4f ms\n", 
                    fabs(tr.tv_sec*1000.0+(double)tr.tv_usec/1000.0));
    }

    for (int i=0; i<output_size; i++) {
      fout << output_base[i] << "\n";
    }
    fout.close();
    
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

    free(output_base);
    free(output_test);
    free(data_samples);
    free(centers);
/*
    delete [] output_base;
    delete [] output_test;
    delete [] data_samples;
    delete [] centers;
    */
  } 
  catch (std::runtime_error &e) {
    printf("%s\n", e.what());
    return -1;
  }

  return 0;
}
