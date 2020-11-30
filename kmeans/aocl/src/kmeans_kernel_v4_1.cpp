#include <string.h>
//#include <cmath>
#include <math.h>
//#include <cfloat>
#include <float.h>
//#include <algorithm>
//#include <sstream>
//#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "kmeans.h"

inline void axpy(double alpha, double* v1, double* v2, int n) {
  //#pragma ACCEL parallel factor=8 
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
/*
inline double dist(double* v1, double* v2, int n) {
  double res = 0.0;
  #pragma ACCEL parallel factor=16
  for (int k=0; k<784; k++) { // n hardcoded - WORKAROUND
    res += (v1[k]-v2[k])*(v1[k]-v2[k]);
  }
  return res;
}
*/
/*
// manual reduction
inline double dist(double* v1, double* v2, int n) {
  const int N = 16;
  double res[N];                                                  
  for (int i=0;i<N;i++) {
      res[i] = 0.0;
  }
  for (int k=0; k<784/N; k++) { 
  #pragma ACCEL parallel
    for (int k_sub=0; k_sub<N; k_sub++) { 
      res[k_sub]+= (v1[k*N+k_sub] - v2[k*N+k_sub]) * (v1[k*N+k_sub] - v2[k*N+k_sub]);
    }
  }
  
  double dist = ((double)0);
  for (int i=0; i<N; i++) {
      dist+= res[i];
  }
  return dist;
}
*/

//#define II_CYCLES 12
//#define DIST_UR 1
//inline double dist(double* v1, double* v2, int n) {
//  double shr[DIST_UR][II_CYCLES+1];
//  for (int k=0; k<DIST_UR; k++) {
//    for (int i=0; i<II_CYCLES+1; i++) {
//      shr[k][i] = ((double)0);
//    }
//  }
//
//  for (int k=0; k<784/DIST_UR; k++) {
//    #pragma ACCEL parallel
//    for (int k_sub=0; k<DIST_UR; k++) {
//      shr[k_sub][II_CYCLES] = shr[k_sub][0] + (v1[k*DIST_UR+k_sub]-v2[k*DIST_UR+k_sub])*(v1[k*DIST_UR+k_sub]-v2[k*DIST_UR+k_sub]);
//
//      #pragma ACCEL parallel
//      for (int j=0; j<II_CYCLES; j++){
//        shr[k_sub][j] = shr[k_sub][j+1];
//      }
//    }
//    /*for (int m=1; m<DIST_UR; m++){
//      for (int n=0; n<DIST_UR; n++){
//        shr[0][II_CYCLES] = shr[0][0] + shr[m][n];
//      }
//    }
//    #pragma ACCEL parallel
//    for (int j=0; j<II_CYCLES; j++){
//      shr[0][j] = shr[0][j+1];
//    }*/
//  }
//
//  double res = 0.0;
//  #pragma ACCEL parallel
//  for (int k=0; k<II_CYCLES; k++) {
//    res += shr[0][k];
//  }
//  return res;
//}

#define II_CYCLES 12
#define DIST_UR 8 
inline double dist(double* v1, double* v2, int n) {
  double shr[DIST_UR][II_CYCLES+1];
  double tmp[DIST_UR];
  for (int k=0; k<DIST_UR; k++) {
    for (int i=0; i<II_CYCLES+1; i++) {
      shr[k][i] = ((double)0);
    }
    tmp[k] = ((double)0);
  }

  for (int k=0; k<784/DIST_UR; k++) { // n hardcoded - WORKAROUND
    #pragma ACCEL parallel
    for (int k_sub=0; k_sub<DIST_UR; k_sub++) {
      shr[k_sub][II_CYCLES] = shr[k_sub][0] + (v1[k*DIST_UR+k_sub]-v2[k*DIST_UR+k_sub])*(v1[k*DIST_UR+k_sub]-v2[k*DIST_UR+k_sub]);

      #pragma ACCEL parallel
      for (int j=0; j<II_CYCLES; j++){
        shr[k_sub][j] = shr[k_sub][j+1];
      }
    }
  }
  
  #pragma ACCEL parallel
  for (int k_sub=0; k_sub<DIST_UR; k_sub++) {
    #pragma ACCEL parallel
    for (int j=0; j<II_CYCLES; j++){
      tmp[k_sub] += shr[k_sub][j];
    }
  }

  double res = 0.0;
  #pragma ACCEL parallel
  for (int j=0; j<DIST_UR; j++){
    res += tmp[j];
  }
  return res;
}
inline double fastSquareDistance(
    double* v1, double norm1,
    double* v2, double norm2,
    int n) 
{
  double sqDist = 0.0;
#ifdef INCLUDE_NORM
  double sumSquaredNorm = norm1 * norm1 + norm2 * norm2;
  double normDiff = norm1 - norm2;

  double precisionBound1 = 2.0 * DBL_EPSILON * sumSquaredNorm / 
    (normDiff * normDiff + DBL_EPSILON);

  double precision = 1e-6;
  if (precisionBound1 < precision) {
    sqDist = sumSquaredNorm - 2.0 * dot(v1, v2, n);
  } 
  // skip Sparse vector case
  else 
#else
  {
    sqDist = dist(v1, v2, n);
  }
#endif
  return sqDist;
}

#ifdef __cplusplus
extern "C" {
#endif
#pragma ACCEL kernel name="kmeans_kernel"
void kmeans(
    int num_samples,
    int num_runs,
    int num_clusters,
    int vector_length,
    double* data,
    double* centers,
    double* output,
    int data_size,
    int center_size,
    int output_size) 
{
#pragma ACCEL interface variable=data depth=data_size max_depth=10250048
#pragma ACCEL interface variable=centers depth=center_size max_depth=10304 
#pragma ACCEL interface variable=output depth=output_size max_depth=10304
//#pragma ACCEL interface variable=data depth=10250048
//#pragma ACCEL interface variable=centers depth=10304 
//#pragma ACCEL interface variable=output depth=10304

  // each data items is norm and vector
  int data_length = vector_length + 1;

  //double* sums   = new double[num_runs*num_clusters*vector_length];
  //int*    counts = new int[num_runs*num_clusters];
  //memset(sums, 0, sizeof(double)*(vector_length*num_runs*num_clusters));
  //memset(counts, 0, sizeof(int)*(num_runs*num_clusters));
  //double* sums = (double *) malloc(sizeof(double)*MAX_NUM_RUNS*MAX_NUM_CLUSTERS*D);
 
  double sums[MAX_NUM_RUNS*MAX_NUM_CLUSTERS*D];
  int counts[MAX_NUM_RUNS*MAX_NUM_CLUSTERS];
  for (int i=0;i<MAX_NUM_RUNS*MAX_NUM_CLUSTERS*D;i++){
      sums[i] = 0;
  }
  for (int i=0;i<MAX_NUM_RUNS*MAX_NUM_CLUSTERS;i++){
      counts[i] = 0;
  }

  assert(num_samples>0 && num_samples<=MAX_NUM_SAMPLES);
  assert(num_runs>0 && num_runs<2);
  assert(num_clusters>0 && num_clusters<=MAX_NUM_CLUSTERS);
  assert(vector_length>0 && vector_length<=D);
  //vector_length = 784;

  // compute sum of centers and counts
  #pragma ACCEL pipeline 
  for (int i=0; i<num_samples; i++) {
#ifdef INCLUDE_NORM		      
      double point_norm = data[i*data_length+data_length];
#else
      double point_norm = 0;
#endif
      double *point = data + i*data_length;

      //for (int r=0; r<num_runs; r++) {
          int r=0;
	  int bestCenter = 0;
          double distance[MAX_NUM_CLUSTERS];

          #pragma ACCEL pipeline
	  for (int k=0; k<num_clusters; k++) {
	      int offset = /*r*num_clusters*data_length +*/ k*data_length;
	      double* center = centers + offset;
              double center_norm = 0;
#ifdef INCLUDE_NORM		      
	      center_norm = centers[offset+data_length];
	      double lowerBoundOfSqDist = center_norm - point_norm;
	      lowerBoundOfSqDist = lowerBoundOfSqDist * lowerBoundOfSqDist;

	      if (lowerBoundOfSqDist < bestDistance) 
#endif
		  {
		      distance[k] = fastSquareDistance(
							   center, center_norm,
							   point, point_norm, 
							   (int)784/*vector_length*/);
		  }
	  }
          
          //double bestDistance = std::numeric_limits<double>::infinity();
	  double bestDistance = INFINITY;
	  for (int k=0; k<num_clusters; k++) {
              if (distance[k] < bestDistance) {
                  bestDistance = distance[k];
                  bestCenter = k;
              }
          }
	  // update sums(r)(bestCenter)
	  double* sum = sums + /*r*num_clusters*vector_length +*/ bestCenter*vector_length;
	  axpy(1.0, point, sum, vector_length);

	  // update counts(r)(bestCenter)
	  counts[/*r*num_clusters +*/ bestCenter] ++;
      //}
  }

  // pack output data
  //for (int i=0; i<num_runs; i++) {
    double i=((double)0);
    for (int j=0; j<num_clusters; j++) {
      double* sum = sums + /*i*num_clusters*vector_length +*/ j*vector_length;
      int offset = i*num_clusters*(2+data_length) + j*(2+data_length);

      output[offset + 0] = i;
      output[offset + 1] = j;
      memcpy(output+offset+2, sum, vector_length*sizeof(double));
      output[offset + 2 + vector_length] = (double)counts[/*i*num_clusters + */j];
    }
  //}

  //free (sums);
  //free (counts);
}
#ifdef __cplusplus
} // extern C
#endif

