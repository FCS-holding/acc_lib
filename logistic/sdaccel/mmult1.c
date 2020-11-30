#include <math.h>
#include <stdio.h>
#include <assert.h>

#define SUPPORT_WIDE_BUS
#ifdef SUPPORT_WIDE_BUS
   #include "ap_cint.h"
   typedef uint512 BTYPE;
   #define      BWIDTH 16 // 16 floats
#else
   typedef float BTYPE;
   #define       BWIDTH  1 
#endif

#define LABEL_SIZE      10
#define FEATURE_SIZE    784
#define FL_SIZE (LABEL_SIZE+FEATURE_SIZE)  // vector size
#define VECS_MAX 800
#define WEIGHTS_SIZE (LABEL_SIZE*(FEATURE_SIZE+1))  //LABEL_SIZE*(FEATURE_SIZE+1)
#define TOTAL_SIZE 60000
#define DATA_SIZE (TOTAL_SIZE*(LABEL_SIZE+FEATURE_SIZE)) //TOTAL_SIZE*(LABEL_SIZE+FEATURE_SIZE)
#if 0
#define CHUNK_SIZE 32
#define PAR 1
#else
#define CHUNK_SIZE 32
#define PAR 2
#endif

void shift_left(float regs[LABEL_SIZE])
{
#pragma HLS INLINE
    int i;
    for(i=0; i<LABEL_SIZE-1; i++)
    {
#pragma HLS UNROLL
        regs[i]=regs[i+1];
    }
}

void recv_data( int exec, BTYPE *global_data, unsigned int offset, float data[CHUNK_SIZE*FL_SIZE] )
{
#pragma HLS INLINE OFF
    if( exec )
    {
      int i, j;

      for (i = 0; i < CHUNK_SIZE*(FL_SIZE)/BWIDTH; i++) {
#pragma HLS pipeline
        BTYPE tmp = global_data[i+offset];
        for (j=0; j<BWIDTH; j++) {
#ifdef SUPPORT_WIDE_BUS
            int ttt = apint_get_range(tmp, (j+1)*32-1, j*32);
            data[i*BWIDTH+j] = *(float*)(&ttt);
#else
            data[i*BWIDTH+j] = tmp;
#endif
        }
      }

//        memcpy( *data, global_data + offset, (LABEL_SIZE+FEATURE_SIZE)*CHUNK_SIZE*4 );
    }
}

void compute1( int exec, float data[CHUNK_SIZE*FL_SIZE], float weights[LABEL_SIZE][FEATURE_SIZE+1], float result1[CHUNK_SIZE][LABEL_SIZE] )
{
#pragma HLS INLINE OFF
    if( exec )
    {
        int i, j, k, p;
        for( j = 0; j < CHUNK_SIZE; j++ )
        {
#pragma HLS PIPELINE
            for( k = 0; k < LABEL_SIZE; k++ )
            {
#pragma HLS UNROLL
                result1[j][k] = weights[k][0];
            }
        }
        for( i = 0; i < FEATURE_SIZE; i++ )
          {
        for( j = 0; j < CHUNK_SIZE/PAR; j++ )
          {
#pragma HLS PIPELINE
        for (p = 0; p < PAR; p++) {
        for( k = 0; k < LABEL_SIZE; k++ )
          {
              result1[PAR*j+p][k] += weights[k][i+1] * data[(PAR*j+p)*FL_SIZE+i+LABEL_SIZE];
          }
          }
          }
          }
    }
}

void compute2( int exec, float result1[CHUNK_SIZE][LABEL_SIZE], float data[CHUNK_SIZE*FL_SIZE], float result2[CHUNK_SIZE][LABEL_SIZE] )
{
#pragma HLS INLINE OFF
    if( exec )
    {
        int j, k;
STAGE2_L: for( k = 0; k < LABEL_SIZE; k++ )
          {
STAGE2_n: for( j = 0; j < CHUNK_SIZE; j++ )
          {
#pragma HLS pipeline
              float temp = 1+result1[j][k]*result1[j][k];
              result2[j][k] = result1[j][k]/temp-(2.f*data[j*FL_SIZE+k]-1.f)/sqrtf(temp);
              //result2[j][k] = 1.f/(1.f+expf(-result1[j][k]))-data[j][k];
          }
          }
    }
}

void compute3( int exec, float result2[CHUNK_SIZE][LABEL_SIZE], float data[CHUNK_SIZE*FL_SIZE], float gradient[LABEL_SIZE][FEATURE_SIZE+1])
{
#pragma HLS INLINE OFF
    float gtmp[PAR][LABEL_SIZE][FEATURE_SIZE+1];
#pragma HLS ARRAY_PARTITION variable="gtmp" complete dim="1"
#pragma HLS ARRAY_PARTITION variable="gtmp" complete dim="2"

    if( exec )
    {
        int i, j, k, p;

        for( j = 0; j < CHUNK_SIZE/PAR; j++ )
          {
        for( i = 0; i <FEATURE_SIZE+1; i++ )
          {
#pragma HLS PIPELINE
        for( k = 0; k < LABEL_SIZE; k++ )
          {
              gradient[k][i] += result2[PAR*j+PAR-1][k] * (i==0 ? 1.f : data[(PAR*j+PAR-1)*FL_SIZE+i-1+LABEL_SIZE]);
              for (p = 0; p < PAR-1; p++)
              {
                float tmp = result2[PAR*j+p][k] * (i==0 ? 1.f : data[(PAR*j+p)*FL_SIZE+i-1+LABEL_SIZE]);
                if (j == 0) gtmp[p][k][i] = tmp;
                else gtmp[p][k][i] += tmp;
              }
          }
          }
          }

        for( i = 0; i <FEATURE_SIZE+1; i++ )
          {
#pragma HLS PIPELINE
        for( k = 0; k < LABEL_SIZE; k++ )
          {
             for (p = 0; p < PAR-1; p++)
             {
              gradient[k][i] += gtmp[p][k][i];
             }
          }
          }

    }
}

void kernel( int i, int n_stage, BTYPE *global_data, 
            float weights[LABEL_SIZE][FEATURE_SIZE+1], 
            float gradient[LABEL_SIZE][FEATURE_SIZE+1], 
            float data_recv[CHUNK_SIZE*FL_SIZE], 
            float data_compute1[CHUNK_SIZE*FL_SIZE], 
            float data_compute2[CHUNK_SIZE*FL_SIZE], 
            float data_compute3[CHUNK_SIZE*FL_SIZE], 
            float result1_prod[CHUNK_SIZE][LABEL_SIZE], 
            float result1_cons[CHUNK_SIZE][LABEL_SIZE], 
            float result2_prod[CHUNK_SIZE][LABEL_SIZE], 
            float result2_cons[CHUNK_SIZE][LABEL_SIZE] 
)
{
#pragma HLS INLINE OFF
    recv_data( i < n_stage, global_data, i * CHUNK_SIZE * (FL_SIZE)/BWIDTH, data_recv );
    compute1( i > 0 && i < n_stage + 1, data_compute1, weights, result1_prod );
    compute2( i > 1 && i < n_stage + 2, result1_cons, data_compute2, result2_prod );
    compute3( i > 2, result2_cons, data_compute3, gradient );
}

void pipeline( int n_samples, BTYPE *global_data, float weights[LABEL_SIZE][FEATURE_SIZE+1], float gradient[LABEL_SIZE][FEATURE_SIZE+1] )
{
    float data0[CHUNK_SIZE*FL_SIZE];
    float data1[CHUNK_SIZE*FL_SIZE];
    float data2[CHUNK_SIZE*FL_SIZE];
    float data3[CHUNK_SIZE*FL_SIZE];
#if 0
#pragma HLS ARRAY_PARTITION variable=data0 cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=data1 cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=data2 cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=data3 cyclic dim=1 factor=8
#endif
    float result10[CHUNK_SIZE][LABEL_SIZE];
    float result11[CHUNK_SIZE][LABEL_SIZE];
    float result20[CHUNK_SIZE][LABEL_SIZE];
    float result21[CHUNK_SIZE][LABEL_SIZE];
#pragma HLS ARRAY_PARTITION variable=result10 complete dim=2
#pragma HLS ARRAY_PARTITION variable=result11 complete dim=2
#pragma HLS ARRAY_PARTITION variable=result20 complete dim=2
#pragma HLS ARRAY_PARTITION variable=result21 complete dim=2

#pragma HLS ARRAY_PARTITION variable=result10 cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=result11 cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=result20 cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=result21 cyclic dim=1 factor=2

    int i;
    int n_stage = n_samples / CHUNK_SIZE;
    int counter = 0;
TOP_LOOP: for(i = 0; i < n_stage + 3; i++ )
          {
              if( counter == 0 )
              {
                  kernel( i, n_stage, global_data, weights, gradient, 
                      data0, data1, data2, data3, 
                      result10, result11, result20, result21 );
              }
              else if ( counter == 1 )
              {
                  kernel( i, n_stage, global_data, weights, gradient, 
                      data3, data0, data1, data2, 
                      result11, result10, result21, result20 );
              }
              else if ( counter == 2 )
              {
                  kernel( i, n_stage, global_data, weights, gradient, 
                      data2, data3, data0, data1, 
                      result10, result11, result20, result21 );
              }
              else if ( counter == 3 )
              {
                  kernel( i, n_stage, global_data, weights, gradient, 
                      data1, data2, data3, data0, 
                      result11, result10, result21, result20 );
              }
              counter++;
              if( counter >= 4 ) counter = 0;
          }
}

void mmult( int n_samples, float global_weights[WEIGHTS_SIZE], BTYPE global_data[DATA_SIZE/BWIDTH], float global_gradient[WEIGHTS_SIZE] )
{
#pragma HLS INTERFACE m_axi port=global_weights offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=global_data offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=global_gradient offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=n_samples bundle=control
#pragma HLS INTERFACE s_axilite port=global_weights bundle=control
#pragma HLS INTERFACE s_axilite port=global_data bundle=control
#pragma HLS INTERFACE s_axilite port=global_gradient bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    float weights[LABEL_SIZE][FEATURE_SIZE+1];
    float gradient[LABEL_SIZE][FEATURE_SIZE+1];
    int i,j,k;
    float regs[LABEL_SIZE];
#pragma HLS ARRAY_PARTITION variable="regs" complete dim="1"
    float buffer[FEATURE_SIZE+1];
#pragma HLS ARRAY_PARTITION variable="weights" complete dim="1"
#pragma HLS ARRAY_PARTITION variable="gradient" complete dim="1"

    n_samples = TOTAL_SIZE;

#if 0
    for(i=0; i < LABEL_SIZE; i++)
    {
#pragma HLS UNROLL
        memcpy((void*)(weights[i]), (const void*)(global_weights+(FEATURE_SIZE+1)*i), (FEATURE_SIZE+1)*4);
        memcpy((void*)(gradient[i]), (const void*)(global_gradient+(FEATURE_SIZE+1)*i), (FEATURE_SIZE+1)*4);
    }
#else
    for(i=0; i < LABEL_SIZE; i++)
    {
#pragma HLS UNROLL
        int j;
        for(j = 0; j < FEATURE_SIZE+1; j++) {
            gradient[i][j] = 0.f;
        }
    }
    for(i=0; i < LABEL_SIZE; i++)
    {
        memcpy((void*)buffer, (const void*)(global_weights+(FEATURE_SIZE+1)*i), (FEATURE_SIZE+1)*4);
        for(j=0; j < FEATURE_SIZE+1; j++)
        {
            for(k=0; k < LABEL_SIZE; k++)
            {
#pragma HLS UNROLL
                regs[k]=weights[k][j];
            }
            shift_left(regs);
            for(k=0; k < LABEL_SIZE; k++)
            {
#pragma HLS UNROLL
                weights[k][j]=regs[k];
            }
            weights[LABEL_SIZE-1][j]=buffer[j];
        }
    }
#endif
    pipeline( n_samples, global_data, weights, gradient);
#if 0
    for(i=0; i < LABEL_SIZE; i++)
    {
#pragma HLS UNROLL
        memcpy((void*)(global_gradient+(FEATURE_SIZE+1)*i), (const void*)(gradient[i]), (FEATURE_SIZE+1)*4);
    }
#else
    for(i=0; i < LABEL_SIZE; i++)
    {
        for(j=0; j < FEATURE_SIZE+1; j++)
        {
            buffer[j]=gradient[0][j];
            for(k=0; k < LABEL_SIZE; k++)
            {
#pragma HLS UNROLL
                regs[k]=gradient[k][j];
            }
            shift_left(regs);
            for(k=0; k < LABEL_SIZE; k++)
            {
#pragma HLS UNROLL
                gradient[k][j]=regs[k];
            }
        }
        memcpy((void*)(global_gradient+i*(FEATURE_SIZE+1)), (const void*)buffer,(FEATURE_SIZE+1)*4);
    }
#endif
}



