#define NUM_THREADS 256

inline void reduction(
    __local double* partial_sums) 
{
  int tid = get_local_id(0);
  int gdim = get_local_size(0);

  barrier(CLK_LOCAL_MEM_FENCE);

  for(int i = gdim/2; i>0; i >>= 1) {
    if(tid < i) {
      partial_sums[tid] += partial_sums[tid + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

inline double vector_dot(
    __global double* v1, 
    __global double* v2, 
    int n) 
{
  __local double partial_sums[NUM_THREADS];

	int tid = get_local_id(0);
	int tdim = get_local_size(0);

  double res = 0.0;
  for (int k=tid; k<n; k+=tdim) {
    res += v1[k]*v2[k];
  }
  partial_sums[tid] = res;

  reduction(partial_sums);

  return partial_sums[0];
}

inline void axpy(
    double alpha, 
    __global double* v1, 
    __global double* v2, 
    int n) 
{
	int tid = get_local_id(0);
	int tdim = get_local_size(0);

  for (int k=tid; k<n; k+=tdim) {
    v2[k] += alpha*v1[k];
  }
}

inline double vector_dist(
    __global double* v1, 
    __global double* v2, 
    int n) 
{
  __local double partial_sums[NUM_THREADS];

	int tid = get_local_id(0);
	int tdim = get_local_size(0);

  double res = 0.0;
  for (int k=tid; k<n; k+=tdim) {
    res += (v1[k]-v2[k])*(v1[k]-v2[k]);
  }

  partial_sums[tid] = res;

  reduction(partial_sums);

  return partial_sums[0];
}

inline double fastSquareDistance(
    __global double* v1, double norm1,
    __global double* v2, double norm2,
    int n)
{

  double precision = 1e-6;
  double sumSquaredNorm = norm1 * norm1 + norm2 * norm2;
  double normDiff = norm1 - norm2;
  double sqDist = 0.0;

  double precisionBound1 = 2.0 * DBL_EPSILON * sumSquaredNorm / 
    (normDiff * normDiff + DBL_EPSILON);

  //if (precisionBound1 < precision) {
  //  sqDist = sumSquaredNorm - 2.0 * vector_dot(v1, v2, n);
  //} 
  //// skip Sparse vector case
  //else 
  {
    sqDist = vector_dist(v1, v2, n);
  }
  return sqDist;
}

__kernel
void kmeans(
		int num_samples,
		int num_runs,
		int num_clusters,
		int vector_length,
		__global double* data, 
		__global double* centers,
		__global double* output)
{
  int gid = get_group_id(0);
	int gdim = get_num_groups(0);

	int tid = get_local_id(0);
	int tdim = get_local_size(0);

  const int data_length = vector_length + 1;
  const int output_size = (vector_length + 3)*num_clusters*num_runs;

  int l_counts[64];

  for (int i=0; i<64; i++) {
    l_counts[i] = 0;
  }

  // memset(output, 0)
  for (int i=0; i<num_runs; i++) {
    for (int j=0; j<num_clusters; j++) {
      int offset = gid*output_size + 
                   i*num_clusters*(2+data_length) + 
                   j*(2+data_length);

      for (int k=tid; k<data_length+2; k+=tdim) {
        output[offset + k] = 0.0;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i=gid; i<num_samples; i+=gdim) {

    double point_norm = data[i*data_length+data_length];
    __global double *point = data + i*data_length;

    for (int r=0; r<num_runs; r++) {
      int bestCenter = 0;
      double bestDistance = DBL_MAX;

      for (int k=0; k<num_clusters; k++) {

        int center_offset = r*num_clusters*data_length 
                          + k*data_length;

        double center_norm = centers[center_offset+data_length];
        __global double* center = centers + center_offset;

        double lowerBoundOfSqDist = center_norm - point_norm;
        lowerBoundOfSqDist = lowerBoundOfSqDist * lowerBoundOfSqDist;

        if (lowerBoundOfSqDist < bestDistance) 
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

      if (tid==0) {
        l_counts[r*num_clusters + bestCenter] ++;
      }

      int output_offset = gid*output_size
                        + r*num_clusters*(2+data_length) 
                        + bestCenter*(2+data_length);

      // update sums(r)(bestCenter)
      __global double* sum = output + output_offset + 2;

      barrier(CLK_LOCAL_MEM_FENCE);

      axpy(1.0, point, sum, vector_length);
    }     
  }

  if (tid==0) {
    for (int i=0; i<num_runs; i++) {
      for (int j=0; j<num_clusters; j++) {
        int offset = gid*output_size +
                     i*num_clusters*(2+data_length) + 
                     j*(2+data_length);

        //output[offset + 0] = i;
        //output[offset + 1] = j;
        output[offset + 2 + vector_length] = 
            (double)l_counts[i*num_clusters + j];
      }
    }
  }
}

__kernel
void vector_sum(
    int num_runs,
    int num_clusters,
		int feature_size,
    int num_vectors,
    __global double* in,
    __global double* out)
{
	int tid = get_local_id(0);
	int tdim = get_local_size(0);

  int vector_size = num_runs*num_clusters*(feature_size+3);

  double results[32];

  for (int k=0; k<vector_size; k+=32*tdim) {
    for (int i=0; i<num_vectors; i++) {
      for (int j=tid; j<vector_size-k && j<32*tdim; j+=tdim) {
        if (i==0) {
          results[j/tdim]  = in[i*vector_size+k+j];
        } else {
          results[j/tdim] += in[i*vector_size+k+j];
        }
      }
    }
    for (int j=tid; j<vector_size-k && j<32*tdim; j+=tdim) {
      out[k+j] = results[j/tdim];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // put correct index on the label
  for (int i=0; i<num_runs; i++) {
    for (int j=tid; j<num_clusters; j+=tdim) {
        int offset = i*num_clusters*(3+feature_size) + j*(3+feature_size);
        out[offset + 0] = i;
        out[offset + 1] = j;
    }
  }
}
