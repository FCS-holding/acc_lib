#include "altera_const.cl"
#pragma OPENCL EXTENSION cl_altera_channels : enable

// Original: #pragma ACCEL array_partition type=channel variable=ch_centers_buf_2 dim=2 factor=4
// Original: #pragma ACCEL array_partition type=channel variable=ch_centers_buf_2 dim=1 factor=10
// Original: #pragma ACCEL array_partition type=channel variable=ch_point_buf_1 dim=2 factor=4
// Original: #pragma ACCEL array_partition type=channel variable=ch_point_buf_1 dim=1 factor=10
// Original: #pragma ACCEL array_partition type=channel variable=ch_distance_0 dim=1 factor=10
 


channel double ch_centers_buf_2_p[10][4] __attribute__((depth(196)));




channel double ch_point_buf_1_p[10][4] __attribute__((depth(196)));




channel double ch_distance_0_p[10] __attribute__((depth(1)));







// Original: #pragma ACCEL kernel name="kmeans_kernel"

 __attribute__ (( autorun))
 __attribute__ (( num_compute_units(10)))
 __attribute__ (( max_global_work_dim(0)))

__kernel void msm_node_0()
{
  int id_0 = get_compute_id(0);
  while(1){
    double distance_sn_tmp_0;
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
/*register*/
    double shr[4][13];
    double tmp[4];
    for (int j = 0; j < 4; j++) {
      
#pragma unroll
      for (int j0 = 0; j0 < 12 + 1; j0++) {
        shr[j][j0] = ((double )0);
      }
      tmp[j] = ((double )0);
    }
// n hardcoded - WORKAROUND
    for (int k0 = 0; k0 < 784 / 4; k0++) {
      
#pragma unroll
      for (int k_sub = 0; k_sub < 4; k_sub++) 
// Original: #pragma ACCEL parallel
// Original: #pragma ACCEL PARALLEL COMPLETE
{
        double centers_buf_sn_tmp_0;
        double point_buf_sn_tmp_0;
        double diff;
        point_buf_sn_tmp_0 = ((double )(read_channel_altera(ch_point_buf_1_p[id_0][k_sub])));
        centers_buf_sn_tmp_0 = ((double )(read_channel_altera(ch_centers_buf_2_p[id_0][k_sub])));
        diff = centers_buf_sn_tmp_0 - point_buf_sn_tmp_0;
        shr[k_sub][12] = shr[k_sub][0] + diff * diff;
        
#pragma unroll
        for (int j = 0; j < 12; j++) 
// Original: #pragma ACCEL parallel
// Original: #pragma ACCEL PARALLEL COMPLETE
{
          shr[k_sub][j] = shr[k_sub][1 + j];
        }
      }
    }
    
#pragma unroll
    for (int k_sub = 0; k_sub < 4; k_sub++) 
// Original: #pragma ACCEL parallel
// Original: #pragma ACCEL PARALLEL COMPLETE
{
      
#pragma unroll
      for (int j = 0; j < 12; j++) 
// Original: #pragma ACCEL parallel
// Original: #pragma ACCEL PARALLEL COMPLETE
{
        tmp[k_sub] += shr[k_sub][j];
      }
    }
    double res;
    res = 0.0;
    
#pragma unroll
    for (int j = 0; j < 4; j++) 
// Original: #pragma ACCEL parallel
// Original: #pragma ACCEL PARALLEL COMPLETE
{
      res += tmp[j];
    }
    distance_sn_tmp_0 = res;
    write_channel_altera(ch_distance_0_p[id_0],distance_sn_tmp_0);
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

void msm_port_point_buf_msm_node_0_0(double point_buf[10][256][4],int k)
{
  int _k;
  _k = k;
// n hardcoded - WORKAROUND
  for (int k0 = 0; k0 < 784 / 4; k0++) {
    
#pragma unroll
    for (int k_sub = 0; k_sub < 4; k_sub++) 
// Original: #pragma ACCEL parallel
// Original: #pragma ACCEL PARALLEL COMPLETE
{
      int k_0;
{
        
#pragma unroll
        for (k_0 = ((int )0); k_0 <= ((int )9); ++k_0) {
          double point_buf_sp_tmp_0;
          point_buf_sp_tmp_0 = point_buf[k_0][k0][k_sub];
          write_channel_altera(ch_point_buf_1_p[k_0][k_sub],point_buf_sp_tmp_0);
        }
      }
    }
  }
}

void msm_port_centers_buf_msm_node_0_0(double centers_buf[10][256][4],int k)
{
  int _k;
  _k = k;
// n hardcoded - WORKAROUND
  for (int k0 = 0; k0 < 784 / 4; k0++) {
    
#pragma unroll
    for (int k_sub = 0; k_sub < 4; k_sub++) 
// Original: #pragma ACCEL parallel
// Original: #pragma ACCEL PARALLEL COMPLETE
{
      int k_0;
{
        
#pragma unroll
        for (k_0 = ((int )0); k_0 <= ((int )9); ++k_0) {
          double centers_buf_sp_tmp_0;
          centers_buf_sp_tmp_0 = centers_buf[k_0][k0][k_sub];
          write_channel_altera(ch_centers_buf_2_p[k_0][k_sub],centers_buf_sp_tmp_0);
        }
      }
    }
  }
}

void msm_port_k_msm_node_0_0(int k)
{
  int _k;
  _k = k;
  _k;
}

__kernel void kmeans(int num_samples,int num_runs,int num_clusters,int vector_length,__global volatile double * restrict data,__global volatile double * restrict centers,__global double * restrict merlin_output,int data_size,int center_size,int output_size)
{







  int __CMOST_KERNEL_ENTRY__kmeans;
  int i_sub_sub;
  int i0_sub_0;
  int j0_sub_sub;
  
#pragma ACCEL interface variable=data burst_off max_depth=10250048 depth=10250048
  
#pragma ACCEL interface variable=centers burst_off max_depth=10304 depth=10304
  
#pragma ACCEL interface burst_off max_depth=10304 depth=10304 variable=merlin_output
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
  double sums[10][64][16];
  int counts[1 * 10];
  for (int i = 0; i < 10; i++) {
    for (int i_sub0 = 0; i_sub0 < 1024 / 16; i_sub0++) {
      
#pragma unroll
      for (int i_sub1 = 0; i_sub1 < 16; i_sub1++) {
        sums[i][i_sub0][i_sub1] = ((double )0);
      }
    }
  }
  
#pragma unroll
  for (int i = 0; i < 1 * 10; i++) {
    counts[i] = 0;
  }
  double centers_buf[10][256][4];
  for (int i = 0; i < num_clusters; i++) {
// Original pramga:  ACCEL PARALLEL FACTOR=8 
    for (int i_sub = 0; i_sub < vector_length / 4 / 8; i_sub++) 
// Original: #pragma ACCEL parallel factor=8
// Original: #pragma ACCEL PARALLEL FACTOR=8
{
      
#pragma unroll
      for (i_sub_sub = 0; i_sub_sub < 8; ++i_sub_sub) 
// Original: #pragma ACCEL parallel factor=8
// Original: #pragma ACCEL PARALLEL FACTOR=8
{
        
#pragma unroll
        for (int i_sub0 = 0; i_sub0 < 4; i_sub0++) {
          centers_buf[i][i_sub * 8 + i_sub_sub][i_sub0] = centers[i * (vector_length + 1) + (i_sub * 8 + i_sub_sub) * 4 + i_sub0];
        }
      }
    }
{
// Original pramga:  ACCEL PARALLEL FACTOR=8 
      for (int i_sub = vector_length / 4 / 8 * 8; i_sub < vector_length / 4; i_sub++) 
// Original: #pragma ACCEL parallel factor=8
// Original: #pragma ACCEL PARALLEL FACTOR=8
{
        
#pragma unroll
        for (int i_sub0 = 0; i_sub0 < 4; i_sub0++) {
          centers_buf[i][i_sub][i_sub0] = centers[i * (vector_length + 1) + i_sub * 4 + i_sub0];
        }
      }
    }
  }
// compute sum of centers and counts
  
#pragma max_concurrency 4
  for (int i = 0; i < num_samples; i++) {
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
    double point_buf[10][256][4];
    for (int i0 = 0; i0 < vector_length / 4; i0++) {
      for (int i1 = 0; i1 < 4; i1++) {
/*num_clusters*/
        
#pragma unroll
        for (int k = 0; k < 10; k++) 
// Original: #pragma ACCEL parallel
// Original: #pragma ACCEL PARALLEL COMPLETE
{
          point_buf[k][i0][i1] = pbuf[i0 * 4 + i1];
        }
      }
    }
//for (int r=0; r<num_runs; r++) {
    int bestCenter;
    bestCenter = 0;
//double bestDistance = std::numeric_limits<double>::infinity();
    double bestDistance;
    bestDistance = ((double )(__builtin_inff()));
    double distance[10];
/*num_clusters*/
    msm_port_point_buf_msm_node_0_0(point_buf,0);
    msm_port_centers_buf_msm_node_0_0(centers_buf,0);
    msm_port_k_msm_node_0_0(0);
    
mem_fence(CLK_CHANNEL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    msm_port_distance_msm_node_0_0(distance,0);
// Original: #pragma ACCEL parallel
// Original: #pragma ACCEL PARALLEL COMPLETE
    for (int k = 0; k < num_clusters; k++) {
      if (distance[k] < bestDistance) {
        bestDistance = distance[k];
        bestCenter = k;
      }
    }
// update sums(r)(bestCenter)
//double** sum = sums[bestCenter];
    double point_1_buf[64][16];
    for (int i0 = 0; i0 < vector_length / 16; i0++) {
      
#pragma unroll
      for (int i0_sub = 0; i0_sub < 16; i0_sub++) {
        point_1_buf[i0][i0_sub] = data[i * (vector_length + 1) + i0 * 16 + i0_sub];
      }
    }
//axpy(1.0, point_1_buf, sum, vector_length);
    double alpha;
    alpha = 1.0;
    for (int k = 0; k < 784 / 16; k++) {
      
#pragma unroll
      for (int k_sub = 0; k_sub < 16; k_sub++) 
// Original: #pragma ACCEL parallel
// Original: #pragma ACCEL PARALLEL COMPLETE
{
        sums[bestCenter][k][k_sub] += alpha * point_1_buf[k][k_sub];
      }
    }
// update counts(r)(bestCenter)
    counts[bestCenter]++;
//}
  }
// pack output data
//for (int i=0; i<num_runs; i++) {
  int i;
  i = 0;
  for (int j = 0; j < num_clusters; j++) {
//double* sum = sums[j];
    int offset;
    offset = j * (2 + data_length);
    merlin_output[offset + 0] = ((double )i);
    merlin_output[offset + 1] = ((double )j);
//memcpy(output+offset+2, sum, vector_length*sizeof(double));
    for (int j0 = 0; j0 < vector_length / 16; j0++) {
// Original pramga:  ACCEL PARALLEL FACTOR=8 
      for (int j0_sub = 0; j0_sub < 2; j0_sub++) 
// Original: #pragma ACCEL parallel factor=8
// Original: #pragma ACCEL PARALLEL FACTOR=8
{
        
#pragma unroll
        for (j0_sub_sub = 0; j0_sub_sub < 8; ++j0_sub_sub) 
// Original: #pragma ACCEL parallel factor=8
// Original: #pragma ACCEL PARALLEL FACTOR=8
{
          merlin_output[offset + 2 + j0 * 16 + (j0_sub * 8 + j0_sub_sub)] = sums[j][j0][j0_sub * 8 + j0_sub_sub];
        }
      }
    }
    merlin_output[offset + 2 + vector_length] = ((double )counts[j]);
  }
//}
//free (sums);
//free (counts);
}
