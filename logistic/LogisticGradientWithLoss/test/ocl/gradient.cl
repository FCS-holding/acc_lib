#define NUM_THREADS 64
#define CHUNK_SIZE  4

inline double reduction(
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

__kernel
void gradient(
		int num_samples,
		int label_size,
		int feature_size,
		__global double* weights,
		__global double* data, 
		__global double* output)
{
	int gid = get_group_id(0);
	int gdim = get_num_groups(0);

	int tid = get_local_id(0);
	int tdim = get_local_size(0);

  const int L = label_size;
  const int D = feature_size; 
  const int weight_size = (L-1)*D;

  // maximum label size = 16
  double margins[16];

  for(int j = tid; j < weight_size+1; j+=tdim) {
    output[gid*(weight_size+1)+j] = 0.0;
  }

  double output_loss = 0.0;

  for(int k = gid; k < num_samples; k+=gdim) {

    double marginY = 0.0;
    double maxMargin = -INFINITY;
    int    maxMarginIndex = 0;

    double  label = data[k*(D+1)];
    __global double* feature = data + k*(D+1) + 1;

    for (int i=0; i<L-1; i++) {

      // dot product
      double margin = vector_dot(weights+i*D, feature, D);

      if (i == (int)label - 1) {
        marginY = margin;
      }
      if (margin > maxMargin) {
        maxMargin = margin;
        maxMarginIndex = i;
      }
      margins[i] = margin;
    }

    double sum = 0.0;
    for (int i=0; i<L-1; i++) {
      if (maxMargin > 0) {
        margins[i] = exp(margins[i]-maxMargin);
        if (i == maxMarginIndex) {
          sum += exp(-maxMargin);
        }
        else {
          sum += margins[i];
        }
      } 
      else {
        margins[i] = exp(margins[i]);
        sum += margins[i];
      }
    }

    // update gradient
    for(int i = 0; i < L-1; i++) {
      __global double* output_group = output + 
                      gid*(weight_size+1) + i*D;

      double multiplier = margins[i] / (sum+1.0);
      if (label != 0.0 && label == i+1) {
        multiplier -= 1.0;
      }

      axpy(multiplier, feature, output_group, D);
    }

    // compute loss
    if (tid==0) {
      double loss = log(sum+1); // math.logip(sum)
      if (label > 0.0) {
        loss -= marginY;
      }
      if (maxMargin > 0) {
        loss += maxMargin;
      }
      output_loss += loss;
    }
  }
  if (tid==0) {
    output[gid*(weight_size+1)+weight_size] = output_loss;
  }
}

__kernel
void vector_sum(
		int vector_size,
    int num_vectors,
    __global double* in,
    __global double* out)
{
	int tid = get_local_id(0);
	int tdim = get_local_size(0);

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
}
