#include "altera_const.cl"
#pragma OPENCL EXTENSION cl_altera_channels : enable

// Original: #pragma ACCEL array_partition type=channel variable=ch_k_5 dim=1 factor=10
// Original: #pragma ACCEL array_partition type=channel variable=ch_centers_buf_4 dim=1 factor=10
// Original: #pragma ACCEL array_partition type=channel variable=ch_centers_buf_3 dim=1 factor=10
// Original: #pragma ACCEL array_partition type=channel variable=ch_point_buf_2 dim=1 factor=10
// Original: #pragma ACCEL array_partition type=channel variable=ch_point_buf_1 dim=1 factor=10
// Original: #pragma ACCEL array_partition type=channel variable=ch_distance_0 dim=1 factor=10
 


channel int ch_k_5_p[10] __attribute__((depth(32)));




channel double ch_centers_buf_4_p[10] __attribute__((depth(784)));




channel double ch_centers_buf_3_p[10] __attribute__((depth(784)));




channel double ch_point_buf_2_p[10] __attribute__((depth(784)));




channel double ch_point_buf_1_p[10] __attribute__((depth(784)));




channel double ch_distance_0_p[10] __attribute__((depth(1)));








static void axpy(double alpha,double *v1,double *v2,int n)
{
//#pragma ACCEL parallel factor=8 
  for (int k = 0; k < n; k++) {
    v2[k] += alpha * v1[k];
  }
}

double dist(int k_0)
{
/*register*/
  double shr[13];
  
#pragma unroll
  for (int i = 0; i < 12 + 1; i++) {
    shr[i] = ((double )0);
  }
// n hardcoded - WORKAROUND
  for (int k = 0; k < 784; k++) {
    double centers_buf_sn_tmp_1;
    double centers_buf_sn_tmp_0;
    double point_buf_sn_tmp_1;
    double point_buf_sn_tmp_0;
    point_buf_sn_tmp_0 = ((double )(read_channel_altera(ch_point_buf_1_p[k_0])));
    point_buf_sn_tmp_1 = ((double )(read_channel_altera(ch_point_buf_2_p[k_0])));
    centers_buf_sn_tmp_0 = ((double )(read_channel_altera(ch_centers_buf_3_p[k_0])));
    centers_buf_sn_tmp_1 = ((double )(read_channel_altera(ch_centers_buf_4_p[k_0])));
    shr[12] = shr[0] + (centers_buf_sn_tmp_0 - point_buf_sn_tmp_0) * (centers_buf_sn_tmp_1 - point_buf_sn_tmp_1);
    
#pragma unroll
    for (int j = 0; j < 12; j++) 
// Original: #pragma ACCEL parallel
// Original: #pragma ACCEL PARALLEL COMPLETE
{
      shr[j] = shr[1 + j];
    }
  }
  double res;
  res = 0.0;
  
#pragma unroll
  for (int j = 0; j < 12; j++) 
// Original: #pragma ACCEL parallel
// Original: #pragma ACCEL PARALLEL COMPLETE
{
    res += shr[j];
  }
  return res;
}

double fastSquareDistance(int k)
{
  double sqDist;
  sqDist = 0.0;
{
    sqDist = dist(k);
  }
  return sqDist;
}

// Original: #pragma ACCEL kernel name="kmeans_kernel"

 __attribute__ (( autorun))
 __attribute__ (( num_compute_units(10)))
 __attribute__ (( max_global_work_dim(0)))

__kernel void msm_node_0()
{
  int id_0 = get_compute_id(0);
  while(1){
    int k_sn_tmp_0;
    int _k;
    k_sn_tmp_0 = ((int )(read_channel_altera(ch_k_5_p[id_0])));
    _k = k_sn_tmp_0;
//int offset = r*num_clusters*data_length + k*data_length;
//double* center = centers_buf[k];
    double center_norm;
    center_norm = ((double )0);
{
      double distance_sn_tmp_0;
      distance_sn_tmp_0 = fastSquareDistance(id_0);
      write_channel_altera(ch_distance_0_p[id_0],distance_sn_tmp_0);
/*vector_length*/
    }
  }
}

void msm_port_distance_msm_node_0_0(double distance[10],int k)
{
  int k_0;
{
    
#pragma unroll
    for (k_0 = ((int )0); k_0 <= ((int )9); ++k_0) {
      double distance_sp_tmp_0;
      int _k;
      _k = k_0;
      distance_sp_tmp_0 = ((double )(read_channel_altera(ch_distance_0_p[k_0])));
      distance[_k] = distance_sp_tmp_0;
    }
  }
}

void msm_port_point_buf_msm_node_0_0(double point_buf[10][1025],int k)
{
  int _k;
  _k = k;
// n hardcoded - WORKAROUND
  for (int k = 0; k < 784; k++) {
    int k_1;
{
      
#pragma unroll
      for (k_1 = ((int )0); k_1 <= ((int )9); ++k_1) {
        double point_buf_sp_tmp_0;
        point_buf_sp_tmp_0 = point_buf[k_1][k];
        write_channel_altera(ch_point_buf_1_p[k_1],point_buf_sp_tmp_0);
      }
    }
  }
// n hardcoded - WORKAROUND
  for (int k = 0; k < 784; k++) {
    int k_0;
{
      
#pragma unroll
      for (k_0 = ((int )0); k_0 <= ((int )9); ++k_0) {
        double point_buf_sp_tmp_1;
        point_buf_sp_tmp_1 = point_buf[k_0][k];
        write_channel_altera(ch_point_buf_2_p[k_0],point_buf_sp_tmp_1);
      }
    }
  }
}

void msm_port_centers_buf_msm_node_0_0(double centers_buf[10][1025],int k)
{
  int _k;
  _k = k;
// n hardcoded - WORKAROUND
  for (int k = 0; k < 784; k++) {
    int k_1;
{
      
#pragma unroll
      for (k_1 = ((int )0); k_1 <= ((int )9); ++k_1) {
        double centers_buf_sp_tmp_0;
        centers_buf_sp_tmp_0 = centers_buf[k_1][k];
        write_channel_altera(ch_centers_buf_3_p[k_1],centers_buf_sp_tmp_0);
      }
    }
  }
// n hardcoded - WORKAROUND
  for (int k = 0; k < 784; k++) {
    int k_0;
{
      
#pragma unroll
      for (k_0 = ((int )0); k_0 <= ((int )9); ++k_0) {
        double centers_buf_sp_tmp_1;
        centers_buf_sp_tmp_1 = centers_buf[k_0][k];
        write_channel_altera(ch_centers_buf_4_p[k_0],centers_buf_sp_tmp_1);
      }
    }
  }
}

void msm_port_k_msm_node_0_0(int k)
{
  int k_0;
{
    
#pragma unroll
    for (k_0 = ((int )0); k_0 <= ((int )9); ++k_0) {
      int k_sp_tmp_0;
      int _k;
      k_sp_tmp_0 = k_0;
      write_channel_altera(ch_k_5_p[k_0],k_sp_tmp_0);
      _k = k_0;
      _k;
    }
  }
}

__kernel void kmeans(int num_samples,int num_runs,int num_clusters,int vector_length,__global volatile double * restrict data,__global volatile double * restrict centers,__global double * restrict merlin_output,int data_size,int center_size,int output_size)
{







  int __CMOST_KERNEL_ENTRY__kmeans;
  int i_sub_sub;
  int i0_sub_0;
  int i0_sub;
  
#pragma ACCEL interface variable=data burst_off max_depth=10250048 depth=10250048
  
#pragma ACCEL interface variable=centers burst_off max_depth=10304 depth=10304
  
#pragma ACCEL interface max_depth=10304 depth=10304 variable=merlin_output
//#pragma ACCEL interface variable=data depth=10250048
//#pragma ACCEL interface variable=centers depth=10304 
//#pragma ACCEL interface variable=output depth=10304
// each data items is norm and vector
  int data_length;
  data_length = vector_length + 1;
//double* sums   = new double[num_runs*num_clusters*vector_length];
//int*    counts = new int[num_runs*num_clusters];
//memset(sums, 0, sizeof(double)*(vector_length*num_runs*num_clusters));
//memset(counts, 0, sizeof(int)*(num_runs*num_clusters));
//double* sums = (double *) malloc(sizeof(double)*MAX_NUM_RUNS*MAX_NUM_CLUSTERS*D);
  double sums[10 * 1024];
  int counts[1 * 10];
  for (int i = 0; i < 1 * 10 * 1024; i++) {
    sums[i] = ((double )0);
  }
  
#pragma unroll
  for (int i = 0; i < 1 * 10; i++) {
    counts[i] = 0;
  }
  double centers_buf[10][1025];
  for (int i = 0; i < num_clusters; i++) {
// Original pramga:  ACCEL PARALLEL FACTOR=8 
    for (int i_sub = 0; i_sub < (1 + vector_length) / 8; i_sub++) 
// Original: #pragma ACCEL parallel factor=8
// Original: #pragma ACCEL PARALLEL FACTOR=8
{
      
#pragma unroll
      for (i_sub_sub = 0; i_sub_sub < 8; ++i_sub_sub) 
// Original: #pragma ACCEL parallel factor=8
// Original: #pragma ACCEL PARALLEL FACTOR=8
{
        centers_buf[i][i_sub * 8 + i_sub_sub] = centers[i * (vector_length + 1) + (i_sub * 8 + i_sub_sub)];
      }
    }
{
// Original pramga:  ACCEL PARALLEL FACTOR=8 
      for (int i_sub = (1 + vector_length) / 8 * 8; i_sub < vector_length + 1; i_sub++) 
// Original: #pragma ACCEL parallel factor=8
// Original: #pragma ACCEL PARALLEL FACTOR=8
{
        centers_buf[i][i_sub] = centers[i * (vector_length + 1) + i_sub];
      }
    }
  }
// compute sum of centers and counts
  #pragma max_concurrency 8
  for (int i = 0; i < num_samples; i++) {
    double point_norm;
    point_norm = ((double )0);
//double *point = data + i*data_length;
    double pbuf[1025];
// Original pramga:  ACCEL PARALLEL FACTOR=8 
    for (int i0 = 0; i0 < (1 + vector_length) / 8; i0++) 
// Original: #pragma ACCEL parallel factor=8
// Original: #pragma ACCEL PARALLEL FACTOR=8
{
      
#pragma unroll
      for (i0_sub_0 = 0; i0_sub_0 < 8; ++i0_sub_0) 
// Original: #pragma ACCEL parallel factor=8
// Original: #pragma ACCEL PARALLEL FACTOR=8
{
        pbuf[i0 * 8 + i0_sub_0] = data[i * (vector_length + 1) + (i0 * 8 + i0_sub_0)];
      }
    }
{
// Original pramga:  ACCEL PARALLEL FACTOR=8 
      for (int i0 = (1 + vector_length) / 8 * 8; i0 < vector_length + 1; i0++) 
// Original: #pragma ACCEL parallel factor=8
// Original: #pragma ACCEL PARALLEL FACTOR=8
{
        pbuf[i0] = data[i * (vector_length + 1) + i0];
      }
    }
    double point_buf[10][1025];
    for (int i0 = 0; i0 < vector_length + 1; i0++) {
/*num_clusters*/
      
#pragma unroll
      for (int k = 0; k < 10; k++) 
// Original: #pragma ACCEL parallel
// Original: #pragma ACCEL PARALLEL COMPLETE
{
        point_buf[k][i0] = pbuf[i0];
      }
    }
//for (int r=0; r<num_runs; r++) {
    int r;
    r = 0;
    int bestCenter;
    bestCenter = 0;
//double bestDistance = std::numeric_limits<double>::infinity();
    double bestDistance;
    bestDistance = ((double )(__builtin_inff()));
    double distance[10];
/*num_clusters*/
// Original pragma: ACCEL STREAM_NODE 
// Original pragma: ACCEL STREAM_NODE PARALLEL_FACTOR=10
    msm_port_point_buf_msm_node_0_0(point_buf,0);
    msm_port_centers_buf_msm_node_0_0(centers_buf,0);
    msm_port_k_msm_node_0_0(0);
    
mem_fence(CLK_CHANNEL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    msm_port_distance_msm_node_0_0(distance,0);
// Original: #pragma ACCEL parallel
// Original: #pragma ACCEL PARALLEL COMPLETE
// Original pragma: ACCEL STREAM_PORT ACCESS_TYPE=read DIM_NUM=2 NODE=msm_node_0 VARIABLE=point_buf
// Original pragma: ACCEL STREAM_PORT ACCESS_TYPE=read DIM_NUM=2 NODE=msm_node_0 VARIABLE=centers_buf
// Original pragma: ACCEL STREAM_PORT ACCESS_TYPE=read DIM_NUM=0 NODE=msm_node_0 VARIABLE=k
    for (int k = 0; k < num_clusters; k++) {
      if (distance[k] < bestDistance) {
        bestDistance = distance[k];
        bestCenter = k;
      }
    }
// update sums(r)(bestCenter)
    double *sum;
    sum = sums + r * num_clusters * vector_length + bestCenter * vector_length;
    double point_1_buf[1025];
// Original pramga:  ACCEL PARALLEL FACTOR=8 
    for (int i0 = 0; i0 < (1 + vector_length) / 8; i0++) 
// Original: #pragma ACCEL parallel factor=8
// Original: #pragma ACCEL PARALLEL FACTOR=8
{
      
#pragma unroll
      for (i0_sub = 0; i0_sub < 8; ++i0_sub) 
// Original: #pragma ACCEL parallel factor=8
// Original: #pragma ACCEL PARALLEL FACTOR=8
{
        point_1_buf[i0 * 8 + i0_sub] = data[i * (vector_length + 1) + (i0 * 8 + i0_sub)];
      }
    }
{
// Original pramga:  ACCEL PARALLEL FACTOR=8 
      for (int i0 = (1 + vector_length) / 8 * 8; i0 < vector_length + 1; i0++) 
// Original: #pragma ACCEL parallel factor=8
// Original: #pragma ACCEL PARALLEL FACTOR=8
{
        point_1_buf[i0] = data[i * (vector_length + 1) + i0];
      }
    }
    axpy(1.0,point_1_buf,sum,vector_length);
// update counts(r)(bestCenter)
    counts[r * num_clusters + bestCenter]++;
//}
  }
// pack output data
//for (int i=0; i<num_runs; i++) {
  int i;
  i = 0;
  for (int j = 0; j < num_clusters; j++) {
    double *sum;
    sum = sums + i * num_clusters * vector_length + j * vector_length;
    int offset;
    offset = i * num_clusters * (2 + data_length) + j * (2 + data_length);
    merlin_output[offset + 0] = ((double )i);
    merlin_output[offset + 1] = ((double )j);
    long long _memcpy_i_0;
    
#pragma unroll 8
    for (_memcpy_i_0 = 0; _memcpy_i_0 < (((unsigned long )vector_length) * 8UL / 8); ++_memcpy_i_0) {
      long long total_offset1;
      total_offset1 = ((long long )(0 * 0 + (offset + 2)));
      long long total_offset2;
      total_offset2 = ((long long )(0 * 0 + 0));
      merlin_output[total_offset1 + _memcpy_i_0] = sum[total_offset2 + _memcpy_i_0];
    }
    merlin_output[offset + 2 + vector_length] = ((double )counts[i * num_clusters + j]);
  }
//}
//free (sums);
//free (counts);
}
