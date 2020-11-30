#include <string.h>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <sstream>
#include <limits>
#include <assert.h>
#include <stdio.h>

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

#define MAX_NUM_RUNS 1

/**
 * stage 0
 * burst input points from DRAM
 */
template <
  unsigned int L,  /* max no, of centers */
  unsigned int D,  /* max point dimension */
  typename DTYPE,  /* compute data type */
  typename BTYPE   /* input bus data type */
  >
void stage_0(
    bool en,                      /* enable signal */
    bool last,                    /* last call */
    int idx,                      /* starting idx for data */ 
    int last_size,                /* size of last chunck */
    int vector_length,            /* input point size */
    DTYPE normv[CHUNK_SIZE],      /* norms of points */
    DTYPE pointv[CHUNK_SIZE][D],  /* point chunk */
    BTYPE* global_data)  /* input bus */
{
#pragma HLS inline off
  if (en) {
    const int data_size = sizeof(DTYPE);
    const int bus_size  = sizeof(BTYPE);
    const int bus_width = bus_size / data_size;

    // load data from memory
    int chunk_idx = 0; 
    int v_idx = 0; 
    int chunk_bound = CHUNK_SIZE;
    
    if (last) chunk_bound = last_size;

    for (int d=0; d<chunk_bound*((vector_length+1+bus_width-1)/bus_width); d++) 
    {
#pragma HLS pipeline
      int offset = idx*CHUNK_SIZE*((vector_length+1+bus_width-1)/bus_width);
      BTYPE tmp_bus = global_data[offset + d];
      int d_idx; // ToDo: merge with v_idx

      for (int j=0; j<bus_width; j++) {
        DTYPE tmp_data;
#ifdef SUPPORT_WIDE_BUS
        ap_uint<data_size*8> tmp_uint = tmp_bus(
            (j+1)*data_size*8-1, j*data_size*8);
        tmp_data = *(DTYPE*)(&tmp_uint);
#else
        tmp_data = tmp_bus;
#endif
        d_idx = v_idx*bus_width+j;
        if (d_idx<vector_length) {
            pointv[chunk_idx][d_idx] = tmp_data;
        } /*else if (d_idx==vector_length) {
            normv[chunk_idx] = tmp_data; 
        } */
      }
      v_idx ++;
      if (v_idx == (vector_length+1+bus_width-1)/bus_width) {
        v_idx = 0;
        chunk_idx ++;
      }
    }
  }
}

/**
 * compute distances
 */
template <
  unsigned int L,  /* max #centers */
  unsigned int D,  /* max point dimension */
  unsigned int F_UR,  /* feature unroll factor */
  typename DTYPE  /* compute data type */
  >
void stage_1(
    bool en,                        /* enable signal */
    int idx,
    int num_clusters,               /* no. of centers */
    int vector_length,              /* point dimension */
    DTYPE centersv[L][D],           /* centers */
    DTYPE pointv[CHUNK_SIZE][D],    /* point chunk */
    DTYPE dist[CHUNK_SIZE][L])      /* distances from points to centers */
{
#pragma HLS inline off
  if (en) {
    DTYPE l_dist[CHUNK_SIZE][L][F_UR];
#pragma HLS ARRAY_PARTITION variable="l_dist" complete dim="2"
#pragma HLS ARRAY_PARTITION variable="l_dist" complete dim="3"

    // initialize 
    for (int k=0; k<CHUNK_SIZE; k++) {
#pragma HLS pipeline
      for (int i=0; i<L; i++) {
        dist[k][i] = 0.0;
        for (int p=0; p<F_UR; p++) {
          l_dist[k][i][p] = 0.0;
        }
      }
    }
    // compute distances
    for (int j=0; j<vector_length; j+=F_UR) {
      for (int k=0; k<CHUNK_SIZE; k++) {
#pragma HLS pipeline
        for (int p=0; p<F_UR; p++) {
          for (int i=0; i<L; i++) {
            double cc = centersv[i][j+p];
            double pp = pointv[k][j+p];
            l_dist[k][i][p] += (cc-pp) * (cc-pp);
          }
        }
      }
    }
    // reduction
    for (int p=0; p<F_UR; p++) {
      for (int k=0; k<CHUNK_SIZE; k++) {
#pragma HLS pipeline
        for (int i=0; i<L; i++) {
          DTYPE dadd_tmp = dist[k][i] + l_dist[k][i][p];
//#pragma HLS resource variable=dadd_tmp core=DADDSUB_nodsp
          dist[k][i] = dadd_tmp;
        }
      }
    }
  }
}

/**
 * Find closest centers
 */
template <
  unsigned int L,  /* max #centers */
  unsigned int D,  /* max point dimension */
  unsigned int F_UR,
  typename DTYPE  /* compute data type */
  >
void stage_2(
    bool en,                        /* enable signal */
    bool last,                      /* last call */
    int idx,
    int last_size,                  /* size of last chunk */
    int num_clusters,               /* no. of centers */
    int vector_length,              /* point dimension */
    DTYPE pointv[CHUNK_SIZE][D],    /* point chunk */
    DTYPE dist[CHUNK_SIZE][L],      /* distances from points to centers */ 
    int bestCenter[CHUNK_SIZE], // to workaround HLS flatten issue
    DTYPE sums[L][D],
    int   counts[L])
{
#pragma HLS inline off
  if (en) {

    int chunk_bound = CHUNK_SIZE;
    DTYPE bestDistance[CHUNK_SIZE];
//    int bestCenter[CHUNK_SIZE];

    if (last) chunk_bound = last_size;

    // initialize 
    for (int k=0; k<chunk_bound; k++) {
#pragma HLS pipeline
      bestCenter[k] = 0;
      bestDistance[k] = std::numeric_limits<double>::infinity();
    }

    // determine best center
    for (int j=0; j<num_clusters; j++) {
      for (int k=0; k<CHUNK_SIZE; k++) {
#pragma HLS pipeline
        DTYPE distance = dist[k][j];
        if (bestDistance[k] > distance) {
            bestDistance[k] = distance;
            bestCenter[k] = j;
        }
      }
    }

    for (int k=0; k<chunk_bound; k++) {
        for (int j=0; j<D; j++) {
#pragma HLS pipeline
#pragma HLS unroll factor=F_UR
#pragma HLS dependence variable=sums inter false // workaround for HLS bug
           sums[bestCenter[k]][j] += pointv[k][j];
        }
    }

    for (int k=0; k<chunk_bound; k++) {
#pragma HLS pipeline II=3
        ++counts[bestCenter[k]];
    }
  }
}

#ifndef HLS_SIM
extern "C" {
#endif
void kmeans(
    int n_samples,
    int num_runs,
    int num_clusters,
    int vector_length,
#if 1
    BUS_TYPE* data,
    double* centers,
    double* output 
#else
    // dimensions for testing -argv "49 9 3"
    BUS_TYPE data[49*((3+8)/8)],
    double centers[9*(3+1)],
    double output[9*(3+3)]
#endif
    )
{
#pragma HLS INTERFACE m_axi port=data offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=centers offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=data bundle=control
#pragma HLS INTERFACE s_axilite port=centers bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control

#pragma HLS INTERFACE s_axilite port=n_samples bundle=control
#pragma HLS INTERFACE s_axilite port=num_runs bundle=control
#pragma HLS INTERFACE s_axilite port=num_clusters bundle=control
#pragma HLS INTERFACE s_axilite port=vector_length bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

 // configurations
  const int L = 10;     // max no. of centers
  const int D = 1024;   // max dimension of a point
  const int F_UR = 4;   // unroll factor for dimension

#ifdef SUPPORT_WIDE_BUS
  const int bus_width = sizeof(BUS_TYPE)/sizeof(double);
#else
  const int bus_width = 1;
#endif

  assert(num_runs == 1);
//num_runs = 1;

#if 0
num_runs = 1;
num_clusters = L;
vector_length = D;
n_samples = 10000;
#endif

  // each data items is norm and vector
  int data_length = vector_length + 1;

  double sums[L][D];
  int    counts[MAX_NUM_RUNS*L];
  int bestCenter[CHUNK_SIZE];

  double centersv[L][D];

  double normv_0[CHUNK_SIZE];
  double normv_1[CHUNK_SIZE];

  double pointv_0[CHUNK_SIZE][D];
  double pointv_1[CHUNK_SIZE][D];
  double pointv_2[CHUNK_SIZE][D];

  double dist_0[CHUNK_SIZE][L];
  double dist_1[CHUNK_SIZE][L];
  double dist_2[CHUNK_SIZE][L];

#if 1
#pragma HLS ARRAY_PARTITION variable="centersv" complete dim="1"
#pragma HLS ARRAY_PARTITION variable="centersv" cyclic factor=F_UR dim="2"
#pragma HLS ARRAY_PARTITION variable="sums" cyclic factor=F_UR dim="2"

#pragma HLS ARRAY_PARTITION variable="pointv_0" cyclic factor=bus_width dim="2"
#pragma HLS ARRAY_PARTITION variable="pointv_1" cyclic factor=bus_width dim="2"
#pragma HLS ARRAY_PARTITION variable="pointv_2" cyclic factor=bus_width dim="2"

#pragma HLS ARRAY_PARTITION variable=dist_0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=dist_1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=dist_2 complete dim=2
#endif

  // initialize the outputs
  for (int i=0; i<L; i++) {
      for (int k=0; k<D; k++) {
#pragma HLS pipeline
#pragma HLS unroll factor=F_UR
	  sums[i][k] = 0.0;
      }
  }
  for (int i = 0; i < num_runs*L; i++) {
#pragma HLS pipeline
      counts[i] = 0;
  }

  // initialize center vectors to 0
  for (int k=0; k<D; k++) {
#pragma HLS pipeline
      for (int i=0; i<L; i++) {
	  centersv[i][k] = 0.0;
      }
  }

  // initialize point vectors to 0
  for (int d=0; d<CHUNK_SIZE; d++) {
      for (int k=0; k<D; k++) {
#pragma HLS pipeline
#pragma HLS unroll factor=bus_width
	  pointv_0[d][k] = 0.0;
	  pointv_1[d][k] = 0.0;
	  pointv_2[d][k] = 0.0;
      }
  }

  // load in centers
  {
      int k=0;
      int j=0;

      for (int d=0; d<num_clusters*data_length; d++) {
#pragma HLS pipeline
	  double val = centers[d];

	  if (j<vector_length) centersv[k][j] = val;
	  j++;
	  if (j==data_length) {j=0; k++;}
      }
  }

 // main computation
  int st3 = 0;
  int upper_bound = (n_samples+CHUNK_SIZE-1)/CHUNK_SIZE;
  int last_chunk_size = n_samples-(upper_bound-1)*CHUNK_SIZE;

  if (last_chunk_size == 0) last_chunk_size = CHUNK_SIZE;
  for(int k=0; k<upper_bound+2; k++) {

    if (st3 == 0) {
      stage_0<L, D>((k<upper_bound), (k == upper_bound-1), k, last_chunk_size, vector_length, normv_0, pointv_0, data);
      stage_1<L, D, F_UR>((k>0 && k<upper_bound+1), k, num_clusters, vector_length, centersv, pointv_1, dist_1);
      stage_2<L, D, F_UR>((k>1), (k == upper_bound+1), k, last_chunk_size, num_clusters, vector_length, pointv_2, dist_2, bestCenter, sums, counts);
    } else if (st3 == 1) {
      stage_0<L, D>((k<upper_bound), (k == upper_bound-1), k, last_chunk_size, vector_length, normv_0, pointv_2, data);
      stage_1<L, D, F_UR>((k>0 && k<upper_bound+1), k, num_clusters, vector_length, centersv, pointv_0, dist_0);
      stage_2<L, D, F_UR>((k>1), (k == upper_bound+1), k, last_chunk_size, num_clusters, vector_length, pointv_1, dist_1, bestCenter, sums, counts);
    } else {
      stage_0<L, D>((k<upper_bound), (k == upper_bound-1), k, last_chunk_size, vector_length, normv_0, pointv_1, data);
      stage_1<L, D, F_UR>((k>0 && k<upper_bound+1), k, num_clusters, vector_length, centersv, pointv_2, dist_2);
      stage_2<L, D, F_UR>((k>1), (k == upper_bound+1), k, last_chunk_size, num_clusters, vector_length, pointv_0, dist_0, bestCenter, sums, counts);
    }

    st3 ++;
    if (st3 == 3) st3 = 0;
    }

  // pack output data
//  for (int i=0; i<num_runs; i++) 
  {// manually flatten the loop nest
    int i = 0;
    int j = 0;
    int k = 0;
    for (int d=0; d<num_clusters*(3 + vector_length); d++) {
#pragma HLS pipeline
        double val;

        if (j == 0) val = i;
        else if (j == 1) val = k;
        else if (j == (2 + vector_length)) val = (double)counts[i*num_clusters + k];
        else val = sums[k][j-2];
        j++;
        if (j == 3 + vector_length) {j = 0; k++;}
        output[d] = val;
    }
  }
}
#ifndef HLS_SIM
} //end of extern "C"
#endif

