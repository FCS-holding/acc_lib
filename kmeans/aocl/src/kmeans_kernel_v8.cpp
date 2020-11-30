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
#define DIST_UR 4 
#define AXPY_UR 16
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
  #pragma ACCEL parallel factor=8
  for (int k=0; k<784; k++) { // n hardcoded - WORKAROUND
    res += (v1[k]-v2[k])*(v1[k]-v2[k]);
  }
  return res;
}
*/

#define II_CYCLES 12
//inline double dist(double* v1, double* v2, int n) {
//  /*register*/ double shr[II_CYCLES+1];
//  for (int i=0; i<II_CYCLES+1; i++) {
//    shr[i] = ((double)0);
//  }
//
//  for (int k=0; k<784; k++) { // n hardcoded - WORKAROUND
//    shr[II_CYCLES] = shr[0] + (v1[k]-v2[k])*(v1[k]-v2[k]);
//
//    #pragma ACCEL parallel
//    for (int j=0; j<II_CYCLES; j++){
//      shr[j] = shr[j+1];
//    }
//  }
//
//  double res = 0.0;
//  #pragma ACCEL parallel
//  for (int j=0; j<II_CYCLES; j++){
//    res += shr[j];
//  }
//  return res;
//}

/*
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
*/
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
#pragma ACCEL interface variable=data depth=data_size burst_off max_depth=10250048
#pragma ACCEL interface variable=centers depth=center_size burst_off max_depth=10304 
#pragma ACCEL interface variable=output depth=output_size burst_off max_depth=10304
//#pragma ACCEL interface variable=data depth=10250048
//#pragma ACCEL interface variable=centers depth=10304 
//#pragma ACCEL interface variable=output depth=10304

  // each data items is norm and vector
  int data_length = vector_length + 1;

  assert(num_samples>0 && num_samples<=MAX_NUM_SAMPLES);
  assert(num_runs>0 && num_runs<2);
  assert(num_clusters>0 && num_clusters<=MAX_NUM_CLUSTERS);
  assert(vector_length>0 && vector_length<=D);
  assert(data_length>0 && data_length<=(D+1));

  //double* sums   = new double[num_runs*num_clusters*vector_length];
  //int*    counts = new int[num_runs*num_clusters];
  //memset(sums, 0, sizeof(double)*(vector_length*num_runs*num_clusters));
  //memset(counts, 0, sizeof(int)*(num_runs*num_clusters));
  //double* sums = (double *) malloc(sizeof(double)*MAX_NUM_RUNS*MAX_NUM_CLUSTERS*D);
 
  double sums[MAX_NUM_CLUSTERS][D/AXPY_UR][AXPY_UR];
  int counts[MAX_NUM_RUNS*MAX_NUM_CLUSTERS];
  for (int i=0;i<MAX_NUM_CLUSTERS;i++){
    for (int i_sub0=0;i_sub0<D/AXPY_UR;i_sub0++){
      for (int i_sub1=0;i_sub1<AXPY_UR;i_sub1++){
        sums[i][i_sub0][i_sub1] = 0;
      }
    }
  }
  for (int i=0;i<MAX_NUM_RUNS*MAX_NUM_CLUSTERS;i++){
      counts[i] = 0;
  }

  double centers_buf[MAX_NUM_CLUSTERS][D/DIST_UR][DIST_UR];
  for (int i=0; i<num_clusters; i++){
    #pragma ACCEL parallel factor=8
    for (int i_sub=0; i_sub<vector_length/DIST_UR; i_sub++){
      for (int i_sub0=0; i_sub0<DIST_UR; i_sub0++){
        centers_buf[i][i_sub][i_sub0] = centers[i*(vector_length+1)+i_sub*DIST_UR+i_sub0];
      }
    }
  }

  // compute sum of centers and counts
  #pragma max_concurrency 4 
  for (int i=0; i<num_samples; i++) {
#ifdef INCLUDE_NORM		      
      double point_norm = data[i*data_length+data_length];
#else
      double point_norm = 0;
#endif
      //double *point = data + i*data_length;
      double pbuf[D+1];
      #pragma ACCEL parallel factor=8
      for (int i0=0; i0<vector_length+1; i0++){
        pbuf[i0] = data[i*(vector_length+1)+i0];
      }
      double point_buf[10][D/DIST_UR][DIST_UR];
      for (int i0=0; i0<vector_length/DIST_UR; i0++){
        for (int i1=0; i1<DIST_UR; i1++){
          #pragma ACCEL parallel
          for (int k=0; k<10/*num_clusters*/; k++) {
            point_buf[k][i0][i1] = pbuf[i0*DIST_UR+i1];
          }
        }
      }

      //for (int r=0; r<num_runs; r++) {
          int r=0;
	  int bestCenter = 0;
	  //double bestDistance = std::numeric_limits<double>::infinity();
	  double bestDistance = INFINITY;
          double distance[MAX_NUM_CLUSTERS];
          #pragma ACCEL parallel
	  for (int k=0; k<10/*num_clusters*/; k++) {
	      //int offset = r*num_clusters*data_length + k*data_length;
	      //double* center = centers_buf[k];
              double center_norm = 0;
#ifdef INCLUDE_NORM		      
	      center_norm = centers[offset+data_length];
	      double lowerBoundOfSqDist = center_norm - point_norm;
	      lowerBoundOfSqDist = lowerBoundOfSqDist * lowerBoundOfSqDist;

	      if (lowerBoundOfSqDist < bestDistance) 
#endif
              {
                  //distance[k] = fastSquareDistance(
                  //                                     centers_buf[k], center_norm,
                  //                                     point_buf[k], point_norm, 
                  //                                     (int)784/*vector_length*/);

                  ///*register*/ double shr[II_CYCLES+1];
                  //for (int j=0; j<II_CYCLES+1; j++) {
                  //  shr[j] = ((double)0);
                  //}
                
                  //for (int k0=0; k0<784; k0++) { // n hardcoded - WORKAROUND
                  //  double diff = centers_buf[k][k0]-point_buf[k][k0];
                  //  shr[II_CYCLES] = shr[0] + diff*diff;
                
                  //  #pragma ACCEL parallel
                  //  for (int j=0; j<II_CYCLES; j++){
                  //    shr[j] = shr[j+1];
                  //  }
                  //}
                
                  //double res = 0.0;
                  //#pragma ACCEL parallel
                  //for (int j=0; j<II_CYCLES; j++){
                  //  res += shr[j];
                  //}
                  //

                  /*register*/ double shr[DIST_UR][II_CYCLES+1];
                  double tmp[DIST_UR];
                  for (int j=0; j<DIST_UR; j++) {
                    for (int j0=0; j0<II_CYCLES+1; j0++) {
                      shr[j][j0] = ((double)0);
                    }
                    tmp[j] = ((double)0);
                  }

                  for (int k0=0; k0<784/DIST_UR; k0++) { // n hardcoded - WORKAROUND
                    #pragma ACCEL parallel
                    for (int k_sub=0; k_sub<DIST_UR; k_sub++) {
                      double diff = centers_buf[k][k0][k_sub]-point_buf[k][k0][k_sub];
                      shr[k_sub][II_CYCLES] = shr[k_sub][0] + diff*diff;

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

                  distance[k] = res;
              }
	  }

	  for (int k=0; k<num_clusters; k++) {
            if (distance[k] < bestDistance) {
                bestDistance = distance[k];
                bestCenter = k;
            }
          }

	  // update sums(r)(bestCenter)
	  //double** sum = sums[bestCenter];
          double point_1_buf[D/AXPY_UR][AXPY_UR];
          for (int i0=0; i0<vector_length/AXPY_UR; i0++){
            for (int i0_sub=0; i0_sub<AXPY_UR; i0_sub++){
              point_1_buf[i0][i0_sub] = data[i*(vector_length+1)+i0*AXPY_UR+i0_sub];
            }
          }
          
	  //axpy(1.0, point_1_buf, sum, vector_length);
          const double alpha = 1.0;
          for (int k=0; k<784/AXPY_UR; k++) {
            #pragma ACCEL parallel 
            for (int k_sub=0; k_sub<AXPY_UR; k_sub++) {
              sums[bestCenter][k][k_sub] += alpha*point_1_buf[k][k_sub];
            }
          }

	  // update counts(r)(bestCenter)
	  counts[bestCenter] ++;
      //}
  }

  // pack output data
  //for (int i=0; i<num_runs; i++) {
    int i=0;
    for (int j=0; j<num_clusters; j++) {
      //double* sum = sums[j];
      int offset = j*(2+data_length);

      output[offset + 0] = i;
      output[offset + 1] = j;
      //memcpy(output+offset+2, sum, vector_length*sizeof(double));
      for (int j0=0; j0<vector_length/AXPY_UR; j0++) {
        #pragma ACCEL parallel factor=8
        for (int j0_sub=0; j0_sub<AXPY_UR; j0_sub++) {
          output[offset+2+j0*AXPY_UR+j0_sub] = sums[j][j0][j0_sub];
        }
      }
      output[offset + 2 + vector_length] = (double)counts[j];
    }
  //}

  //free (sums);
  //free (counts);
}
#ifdef __cplusplus
} // extern C
#endif

