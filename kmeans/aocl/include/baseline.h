#include <string.h>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <sstream>
#include <limits>

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

#ifdef INCLUDE_NORM
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

void kmeans_base(
    int num_samples,
    int num_runs,
    int num_clusters,
    int vector_length,
    double* data,
    double* centers,
    double* output) 
{
  // each data items is norm and vector
  int data_length = vector_length + 1;

  double* sums   = new double[num_runs*num_clusters*vector_length];
  int*    counts = new int[num_runs*num_clusters];

  memset(sums, 0, sizeof(double)*(vector_length*num_runs*num_clusters));
  memset(counts, 0, sizeof(int)*(num_runs*num_clusters));

  // compute sum of centers and counts
  for (int i=0; i<num_samples; i++) {
      double point_norm = data[i*data_length+data_length];
      double *point = data + i*data_length;

      for (int r=0; r<num_runs; r++) {
	  int bestCenter = 0;
	  double bestDistance = std::numeric_limits<double>::infinity();

	  for (int k=0; k<num_clusters; k++) {
	      int offset = r*num_clusters*data_length + k*data_length;
	      double center_norm = centers[offset+data_length];
	      double* center = centers + offset;

	      double lowerBoundOfSqDist = center_norm - point_norm;
	      lowerBoundOfSqDist = lowerBoundOfSqDist * lowerBoundOfSqDist;

#ifdef INCLUDE_NORM		      
	      if (lowerBoundOfSqDist < bestDistance) 
#endif
		  {
		      double distance = fastSquareDistance(
							   center, center_norm,
							   point, point_norm, 
							   vector_length);
		      if (distance < bestDistance) {
			  bestDistance = distance;
			  bestCenter = k;
		      }
		  }
	  }

	  // update sums(r)(bestCenter)
	  double* sum = sums + r*num_clusters*vector_length + bestCenter*vector_length;
	  axpy(1.0, point, sum, vector_length);

	  // update counts(r)(bestCenter)
	  counts[r*num_clusters + bestCenter] ++;
      }
  }

  // pack output data
  for (int i=0; i<num_runs; i++) {
    for (int j=0; j<num_clusters; j++) {
      double* sum = sums + i*num_clusters*vector_length + j*vector_length;
      int offset = i*num_clusters*(2+data_length) + j*(2+data_length);

      output[offset + 0] = i;
      output[offset + 1] = j;
      memcpy(output+offset+2, sum, vector_length*sizeof(double));
      output[offset + 2 + vector_length] = (double)counts[i*num_clusters + j];
    }
  }

  delete sums;
  delete counts;
}
