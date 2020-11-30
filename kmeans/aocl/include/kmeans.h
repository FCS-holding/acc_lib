#define CHUNK_SIZE 16
#define MAX_NUM_RUNS 1
#define MAX_NUM_CLUSTERS 10
#define MAX_NUM_SAMPLES 10000

#ifndef HLS_SIM
//#define SUPPORT_WIDE_BUS
#endif
#ifdef  SUPPORT_WIDE_BUS
#include "ap_int.h"
typedef ap_uint<512> BUS_TYPE;
#else
typedef double BUS_TYPE;
#endif

typedef BUS_TYPE BTYPE;
typedef double DTYPE;
// configurations
//const int L = 10;     // max no. of centers
//const int D = 1024;   // max dimension of a point
//const int F_UR = 4;   // unroll factor for dimensionypedef double DTYPE;
#define L 10     // max no. of centers
#define D 1024   // max dimension of a point
#define F_UR 4   // unroll factor for dimensionypedef double DTYPE;

#ifdef __cplusplus
extern "C" {
#endif
void kmeans(
    int num_samples,
    int num_runs,
    int num_clusters,
    int vector_length,
    BUS_TYPE* data,
    double* centers,
    double* output, 
    int data_size, 
    int center_size, 
    int output_size
    );
#ifdef __cplusplus
}
#endif

