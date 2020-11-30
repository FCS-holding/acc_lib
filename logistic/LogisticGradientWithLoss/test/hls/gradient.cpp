#include <stdio.h>
#include <string.h>
#include <cmath>

#define CHUNK_SIZE 16

#ifndef HLS_SIM
#define SUPPORT_WIDE_BUS
#endif
#ifdef  SUPPORT_WIDE_BUS
#include "ap_int.h"
typedef ap_uint<512> BUS_TYPE;
#else 
typedef double BUS_TYPE;
#endif

/**
 * logistic compute stage 0
 * burst input data from DRAM
 */
template <
  unsigned int L,  /* max label size */
  unsigned int D,  /* max label feature */
  typename DTYPE,  /* feature compute data type */
  typename BTYPE   /* input bus data type */
  >
void stage_0(
    bool en,                      /* enable signal */
    int idx,                      /* starting idx for data */ 
    int n_samples,                /* total data sample size */
    int label_size,               /* input label size */
    int feature_size,             /* input feature size */
    DTYPE label[CHUNK_SIZE],      /* input lable data */
    DTYPE feature[CHUNK_SIZE][D], /* input feature data */
    BTYPE* global_data)           /* input bus */
{
#pragma HLS inline off
  if (en) {
    const int data_size = sizeof(DTYPE);
    const int bus_size  = sizeof(BTYPE);
    const int bus_width = bus_size / data_size;

    // indexes in flattened loop
    int chunk_idx = 0; 
    int feat_idx = 0; 

    int feature_bound = (feature_size+bus_width)/bus_width;
    int chunk_bound = CHUNK_SIZE;
    if ((idx+1)*CHUNK_SIZE > n_samples) {
      chunk_bound = n_samples - idx*CHUNK_SIZE;
    }

    for (int d=0; d<chunk_bound*feature_bound; d++) 
    {
#pragma HLS pipeline
      int offset = idx*CHUNK_SIZE*feature_bound;
      BTYPE tmp_bus = global_data[offset + d];

      for (int j=0; j<bus_width; j++) {
        DTYPE tmp_data;
#ifdef SUPPORT_WIDE_BUS
        ap_uint<data_size*8> tmp_uint = tmp_bus(
            (j+1)*data_size*8-1, j*data_size*8);
        tmp_data = *(DTYPE*)(&tmp_uint);
#else
        tmp_data = tmp_bus;
#endif
        if (feat_idx==0 && j==0) {
          label[chunk_idx] = tmp_data;
        } else if (feat_idx*bus_width+j<feature_size+1) {
          feature[chunk_idx][feat_idx*bus_width+j-1] = tmp_data;
        }
      }
      feat_idx ++;
      if (feat_idx == feature_bound) {
        feat_idx = 0;
        chunk_idx ++;
      }
    }
  }
}

/**
 * logistic compute stage 1
 * dot product to get margins output
 */
template <
  unsigned int L,     /* max label size */
  unsigned int D,     /* max feature size */
  unsigned int F_UR,  /* feature unroll factor */
  typename DTYPE      /* feature compute data type */
  >
void stage_1(
    bool en,                        /* enable signal */
    int label_size,                 /* input label size */
    int feature_size,               /* input feature size */
    DTYPE margins[CHUNK_SIZE][L-1], /* output margins */
    DTYPE weights[L-1][D],          /* input weights */
    DTYPE feature[CHUNK_SIZE][D])   /* input feature */
{
#pragma HLS inline off
  if (en) {
    DTYPE l_margins[CHUNK_SIZE][L-1][F_UR];
#pragma HLS ARRAY_PARTITION variable="l_margins" complete dim="2"
#pragma HLS ARRAY_PARTITION variable="l_margins" complete dim="3"

    // initialize 
    for (int k=0; k<CHUNK_SIZE; k++) {
#pragma HLS pipeline
      for (int i=0; i<L-1; i++) {
        margins[k][i] = 0.0;
        for (int p=0; p<F_UR; p++) {
          l_margins[k][i][p] = 0.0;
        }
      }
    }
    // dot product
    for (int j=0; j<feature_size; j+=F_UR) {
      for (int k=0; k<CHUNK_SIZE; k++) {
#pragma HLS pipeline
        for (int p=0; p<F_UR; p++) {
          for (int i=0; i<L-1; i++) {
            l_margins[k][i][p] += weights[i][j+p] * feature[k][j+p];
          }
        }
      }
    }
    // reduction
    for (int p=0; p<F_UR; p++) {
      for (int k=0; k<CHUNK_SIZE; k++) {
#pragma HLS pipeline
        for (int i=0; i<L-1; i++) {
          DTYPE dadd_tmp = margins[k][i] + l_margins[k][i][p];
//#pragma HLS resource variable=dadd_tmp core=DADDSUB_nodsp
          margins[k][i] = dadd_tmp;
        }
      }
    }
  }
}

/**
 * logistic compute stage 2
 * calculate loss update and multipliers for gradient update
 */
template <
  unsigned int L,     /* max label size */
  unsigned int D,     /* max feature size */
  unsigned int F_UR,  /* feature unroll factor */
  typename DTYPE      /* feature compute data type */
  >
void stage_2(
    bool en,                            /* enable signal */
    int idx,                            /* starting idx for data */ 
    int n_samples,                      /* total data sample size */
    int label_size,                     /* input label size */
    int feature_size,                   /* input feature size */
    DTYPE multipliers[CHUNK_SIZE][L-1], /* output multipliers */
    DTYPE out_loss[CHUNK_SIZE],         /* output loss */
    DTYPE label[CHUNK_SIZE],            /* input label data */
    DTYPE margins[CHUNK_SIZE][L-1])     /* input margins results */
{
#pragma HLS inline off
  if (en) {
    int offset = idx*CHUNK_SIZE;
    double marginY[CHUNK_SIZE];
    double maxMargin[CHUNK_SIZE];
    int    maxMarginIndex[CHUNK_SIZE];

    // initialize local buffers
    for (int k=0; k<CHUNK_SIZE; k++) {
#pragma HLS pipeline
      marginY[k] = 0.0;
      maxMargin[k] = -INFINITY;
      maxMarginIndex[k] = 0;

      if (label[k] > 0) {
        marginY[k] = margins[k][(int)label[k]-1];
      }
      for (int i=0; i<L-1; i++) {
        multipliers[k][i] = 0.0;
      }
    }

    // find maximum margin
    for (int i=0; i<label_size-1; i++) {
      for (int k=0; k<CHUNK_SIZE; k++) {
#pragma HLS pipeline
        if (margins[k][i] > maxMargin[k]) {
          maxMargin[k] = margins[k][i];
          maxMarginIndex[k] = i;
        }
      }
    }

    // calculate exponential multipliers
    for (int k=0; k<CHUNK_SIZE; k++) {
#pragma HLS pipeline II=L-1
      for (int i=0; i<L-1; i++) {
        double tmp;
        if (maxMargin[k] > 0) {
          tmp = margins[k][i]-maxMargin[k];
        }
        else {
          tmp = margins[k][i];
        }
        double tmp_exp = exp(tmp);
//#pragma HLS resource variable=tmp_exp core=dexp_nodsp
        margins[k][i] = tmp_exp;
      }
    }
    double sum[CHUNK_SIZE];

    for (int k=0; k<CHUNK_SIZE; k++) {
#pragma HLS pipeline
      sum[k] = 0.0;
    }

    for (int i=0; i<label_size-1; i++) {
      for (int k=0; k<CHUNK_SIZE; k++) {
#pragma HLS pipeline
        if (i != maxMarginIndex[k]) {
          sum[k] += margins[k][i];
        }
      }
    }
    for (int k=0; k<CHUNK_SIZE; k++) {
#pragma HLS pipeline
      if (maxMargin[k] > 0) {
        double tmp = exp(-maxMargin[k]);
//#pragma HLS resource variable=tmp core=dexp_nodsp
        sum[k] += tmp; 
      }
      else {
        sum[k] += margins[k][maxMarginIndex[k]];
      }

      for (int i=0; i<L-1; i++) {
        multipliers[k][i] = margins[k][i] / (sum[k]+1.0);
        if (label[k] != 0.0 && label[k] == i+1) {
          multipliers[k][i] -= 1.0;
        }
      }
      double loss = out_loss[k];
      double tmp_log = log(sum[k]+1);
//#pragma HLS resource variable=tmp_log core=dlog_nodsp
      loss += tmp_log; // math.logip(sum)
      if (label[k] > 0.0) {
        loss -= marginY[k];
      }
      if (maxMargin[k] > 0) {
        loss += maxMargin[k];
      }
      if (k+offset<n_samples) {
        out_loss[k] = loss;
      }
    }
    for (int k=n_samples-offset; k<CHUNK_SIZE; k++) {
#pragma HLS pipeline
      for (int i=0; i<L-1; i++) {
        multipliers[k][i] = 0.0;
      }
    }
  }
}

/**
 * logistic compute stage 3
 * gradient update with cblas_axpy()
 */
template <
  unsigned int L,     /* max label size */
  unsigned int D,     /* max feature size */
  unsigned int F_UR,  /* feature unroll factor */
  typename DTYPE      /* feature compute data type */
  >
void stage_3(
    bool en,                            /* enable signal */
    int idx,                            /* starting idx for data */ 
    int n_samples,                      /* total data sample size */
    int label_size,                     /* input label size */
    int feature_size,                   /* input feature size */
    DTYPE output[CHUNK_SIZE][L-1][D],   /* output gradients */
    DTYPE multipliers[CHUNK_SIZE][L-1], /* input multipliers */
    DTYPE feature[CHUNK_SIZE][D])       /* input features */ 
{
#pragma HLS inline off
  if (en) {
    // update gradient
    for (int j=0; j<feature_size; j+=F_UR) {
      for (int k=0; k<CHUNK_SIZE; k++) {
#pragma HLS pipeline
        for (int p=0; p<F_UR; p++) {
          for(int i=0; i<L-1; i++) {
#pragma HLS dependence variable=output array inter false
            //output[k][i][j+p] += multipliers[k][i]*feature[k][j+p];
            DTYPE mul_tmp = multipliers[k][i]*feature[k][j+p];
            DTYPE add_tmp = output[k][i][j+p] + mul_tmp;
            output[k][i][j+p] = add_tmp;
          }
        }
      }
    }
  }
}

#ifndef HLS_SIM
extern "C" {
void gradient( 
    int n_samples, 
    int label_size, 
    int feature_size,
    double*   global_weights, 
    BUS_TYPE* global_data, 
    double*   global_output );
}
#endif

void gradient( 
    int n_samples, 
    int label_size, 
    int feature_size,
    double*   global_weights, 
    BUS_TYPE* global_data, 
    double*   global_output)
    //double   global_weights[9*784], 
    //BUS_TYPE global_data[32*785], 
    //double   global_output[9*784+1])
{
#pragma HLS INTERFACE m_axi port=global_weights offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=global_data offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=global_output offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=n_samples bundle=control
#pragma HLS INTERFACE s_axilite port=label_size bundle=control
#pragma HLS INTERFACE s_axilite port=feature_size bundle=control
#pragma HLS INTERFACE s_axilite port=global_weights bundle=control
#pragma HLS INTERFACE s_axilite port=global_data bundle=control
#pragma HLS INTERFACE s_axilite port=global_output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // configurations
  const int L = 10;     // max label size
  const int D = 1024;   // max feature size
  const int F_UR = 4;   // feature unroll factor

  //label_size = L;
  //feature_size = D;

#ifdef SUPPORT_WIDE_BUS
  const int bus_width = sizeof(BUS_TYPE)/sizeof(double);
#else
  const int bus_width = 1;
#endif
 
  double regs[L-1];
  double weights[L-1][D];

  double label_0[CHUNK_SIZE];
  double label_1[CHUNK_SIZE];
  double label_2[CHUNK_SIZE];
  double label_3[CHUNK_SIZE];

  double feature_0[CHUNK_SIZE][D];
  double feature_1[CHUNK_SIZE][D];
  double feature_2[CHUNK_SIZE][D];
  double feature_3[CHUNK_SIZE][D];

  double loss[CHUNK_SIZE];
  double output[CHUNK_SIZE][L-1][D];
  double output_final[L-1][D];

  double margins_0[CHUNK_SIZE][L-1];
  double margins_1[CHUNK_SIZE][L-1];
  double multipliers_0[CHUNK_SIZE][L-1];
  double multipliers_1[CHUNK_SIZE][L-1];

#pragma HLS ARRAY_PARTITION variable="regs" complete dim="1"
#pragma HLS ARRAY_PARTITION variable="weights" complete dim="1"
#pragma HLS ARRAY_PARTITION variable="weights" cyclic factor=F_UR dim="2"

#pragma HLS ARRAY_PARTITION variable="feature_0" cyclic factor=bus_width dim="2"
#pragma HLS ARRAY_PARTITION variable="feature_1" cyclic factor=bus_width dim="2"
#pragma HLS ARRAY_PARTITION variable="feature_2" cyclic factor=bus_width dim="2"
#pragma HLS ARRAY_PARTITION variable="feature_3" cyclic factor=bus_width dim="2"

#pragma HLS ARRAY_PARTITION variable="output" complete dim="2"
#pragma HLS ARRAY_PARTITION variable="output" cyclic factor=F_UR dim="3"

#pragma HLS ARRAY_PARTITION variable="output_final" complete dim="1"
#pragma HLS ARRAY_PARTITION variable="output_final" cyclic factor=F_UR dim="2"

#pragma HLS ARRAY_PARTITION variable="margins_0" complete dim="2"
#pragma HLS ARRAY_PARTITION variable="margins_1" complete dim="2" 

#pragma HLS ARRAY_PARTITION variable="multipliers_0" complete dim="2"
#pragma HLS ARRAY_PARTITION variable="multipliers_1" complete dim="2"

  // initialize output gradients
  for (int k=0; k<D; k++) {
#pragma HLS pipeline
    for (int i=0; i<L-1; i++) {
      output_final[i][k] = 0.0;
      weights[i][k] = 0.0;
    }
  }
  for (int d=0; d<CHUNK_SIZE; d++) {
#pragma HLS pipeline
    loss[d] = 0.0;
  }
  for (int d=0; d<CHUNK_SIZE; d++) {
    for (int k=0; k<D; k++) {
#pragma HLS pipeline
      feature_0[d][k] = 0.0;
      feature_1[d][k] = 0.0;
      feature_2[d][k] = 0.0;
      feature_3[d][k] = 0.0;
      for (int i=0; i<L-1; i++) {
        output[d][i][k] = 0.0;
        output_final[i][k] = 0.0;
      }
    }
  }

  // load weights
  // use shift-regs to workaround hls bugs
  double buffer[D];
  for (int i=0; i<label_size-1; i++) {

    memcpy(
        (void*)buffer, 
        (const void*)(global_weights+i*feature_size), 
        feature_size*sizeof(double));

    for (int j=0; j<feature_size; j++) {
      for (int k=0; k<L-1; k++) {
#pragma HLS unroll
        regs[k] = weights[k][j];
      }
      for (int k=0; k<L-2; k++) {
#pragma HLS unroll
        regs[k] = regs[k+1];
      }
      for (int k=0; k<L-1; k++) {
#pragma HLS unroll
        weights[k][j] = regs[k];
      }
      weights[label_size-2][j] = buffer[j];
    }
  }

  // main computation pipeline
  int stage = 0;
  int upper_bound = (n_samples+CHUNK_SIZE-1)/CHUNK_SIZE;
  for(int k=0; k<upper_bound+3; k++) {

    if (stage == 0) {
      stage_0<L, D>((k<upper_bound), k, n_samples, label_size, feature_size, label_0, feature_0, global_data);
      stage_1<L, D, F_UR>((k>0 && k<upper_bound+1), label_size, feature_size, margins_0, weights, feature_3);
      stage_2<L, D, F_UR>((k>1 && k<upper_bound+2), k-2, n_samples, label_size, feature_size, multipliers_0, loss, label_2, margins_1);
      stage_3<L, D, F_UR>((k>2), k-3, n_samples, label_size, feature_size, output, multipliers_1, feature_1);
    } else if (stage == 1) {
      stage_0<L, D>((k<upper_bound), k, n_samples, label_size, feature_size, label_1, feature_1, global_data);
      stage_1<L, D, F_UR>((k>0 && k<upper_bound+1), label_size, feature_size, margins_1, weights, feature_0);
      stage_2<L, D, F_UR>((k>1 && k<upper_bound+2), k-2, n_samples, label_size, feature_size, multipliers_1, loss, label_3, margins_0);
      stage_3<L, D, F_UR>((k>2), k-3, n_samples, label_size, feature_size, output, multipliers_0, feature_2);
    } else if (stage == 2) {
      stage_0<L, D>((k<upper_bound), k, n_samples, label_size, feature_size, label_2, feature_2, global_data);
      stage_1<L, D, F_UR>((k>0 && k<upper_bound+1), label_size, feature_size, margins_0, weights, feature_1);
      stage_2<L, D, F_UR>((k>1 && k<upper_bound+2), k-2, n_samples, label_size, feature_size, multipliers_0, loss, label_0, margins_1);
      stage_3<L, D, F_UR>((k>2), k-3, n_samples, label_size, feature_size, output, multipliers_1, feature_3);
    } else  {
      stage_0<L, D>((k<upper_bound), k, n_samples, label_size, feature_size, label_3, feature_3, global_data);
      stage_1<L, D, F_UR>((k>0 && k<upper_bound+1), label_size, feature_size, margins_1, weights, feature_2);
      stage_2<L, D, F_UR>((k>1 && k<upper_bound+2), k-2, n_samples, label_size, feature_size, multipliers_1, loss, label_1, margins_0);
      stage_3<L, D, F_UR>((k>2), k-3, n_samples, label_size, feature_size, output, multipliers_0, feature_0);
    }

    stage ++;
    if (stage == 4) stage = 0;
  }

  // reduce local output and loss
  for (int d=0; d<CHUNK_SIZE; d++) {
    for (int k=0; k<D; k+=F_UR) {
#pragma HLS pipeline II=F_UR
      for (int p=0; p<F_UR; p++) {
        for (int i=0; i<L-1; i++) {
          output_final[i][k+p] += output[d][i][k+p];
        }
      }
    }
  }
  for (int d=1; d<CHUNK_SIZE; d++) {
    loss[0] += loss[d];
  }

  // put output data
  for (int k=0; k<(label_size-1)*feature_size+1; k++) {
#pragma HLS pipeline
    double tmp;
    if (k<(label_size-1)*feature_size) {
      tmp = output_final[k/feature_size][k%feature_size];
    } else {
      tmp = loss[0];
    }
    global_output[k] = tmp;
  }
}
