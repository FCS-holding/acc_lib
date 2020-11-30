#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
//#include <CL/opencl.h>
#include <sys/time.h>

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_half.h"

#define EXTN 0

#if EXTN
extern "C" {
#endif

////////////////////////////////////////////////////////////////////////////////
// hardware specification
#define UNROLL 32
#define DSIZE 16
#define HWFIn UNROLL
#define HWFOut UNROLL
#define HWFinR 30
#define HWFinC 226
#define HWFR 28
#define HWFC 224
#define HWKsize 3
#define HWKstride 1

#define HWin HWFIn
#define HWout HWFOut
#define BitW DSIZE*UNROLL

//typedef ap_fixed<32, 16> data_t;
//typedef float data_t;
typedef half data_t;
typedef ap_uint<BitW> bw_t;
//typedef int bw_t;

//typedef int data_t;
#define POOL 1
#define RELU 1
////////////////////////////////////////////////////////////////////////////////
// cnn module specification
#define READ 1
#define WRITE 0

#define DATA_R 224
#define DATA     3
#define CONV11   64
#define CONV12   64
#define POOL12   64
#define CONV21   128
#define CONV22   128
#define POOL22   128
#define CONV31   256
#define CONV32   256
#define CONV33   256
#define POOL33   256
#define CONV41   512
#define CONV42   512
#define CONV43   512
#define POOL43   512
#define CONV51   512
#define CONV52   512
#define CONV53   512
#define POOL53   512
#define PAD      1
#define STRIDE   1
#define KSIZE    3

#define CONV11_I (DATA_R + 2*PAD)
#define CONV11_R ((CONV11_I - KSIZE + STRIDE)/STRIDE)
#define CONV12_I (CONV11_R + 2*PAD)
#define CONV12_R ((CONV12_I - KSIZE + STRIDE)/STRIDE)
#define POOL12_R (CONV12_R/2)
#define CONV21_I (POOL12_R + 2*PAD)
#define CONV21_R ((CONV21_I - KSIZE + STRIDE)/STRIDE)
#define CONV22_I (CONV21_R + 2*PAD)
#define CONV22_R ((CONV22_I - KSIZE + STRIDE)/STRIDE)
#define POOL22_R (CONV22_R/2)
#define CONV31_I (POOL22_R + 2*PAD)
#define CONV31_R ((CONV31_I - KSIZE + STRIDE)/STRIDE)
#define CONV32_I (CONV31_R + 2*PAD)
#define CONV32_R ((CONV32_I - KSIZE + STRIDE)/STRIDE)
#define CONV33_I (CONV32_R + 2*PAD)
#define CONV33_R ((CONV33_I - KSIZE + STRIDE)/STRIDE)
#define POOL33_R (CONV33_R/2)
#define CONV41_I (POOL33_R + 2*PAD)
#define CONV41_R ((CONV41_I - KSIZE + STRIDE)/STRIDE)
#define CONV42_I (CONV41_R + 2*PAD)
#define CONV42_R ((CONV42_I - KSIZE + STRIDE)/STRIDE)
#define CONV43_I (CONV42_R + 2*PAD)
#define CONV43_R ((CONV43_I - KSIZE + STRIDE)/STRIDE)
#define POOL43_R (CONV43_R/2)
#define CONV51_I (POOL43_R + 2*PAD)
#define CONV51_R ((CONV51_I - KSIZE + STRIDE)/STRIDE)
#define CONV52_I (CONV51_R + 2*PAD)
#define CONV52_R ((CONV52_I - KSIZE + STRIDE)/STRIDE)
#define CONV53_I (CONV52_R + 2*PAD)
#define CONV53_R ((CONV53_I - KSIZE + STRIDE)/STRIDE)
#define POOL53_R (CONV53_R/2)

//#define Layer11 (DATA*CONV11*KSIZE*KSIZE+CONV11)

#define LayerDef 80*16
#define Layer11 (UNROLL*CONV11*KSIZE*KSIZE+CONV11)
#define Layer12 (Layer11+ CONV11*CONV12*KSIZE*KSIZE+CONV12)
#define Layer21 (Layer12+ CONV12*CONV21*KSIZE*KSIZE+CONV21)
#define Layer22 (Layer21+ CONV21*CONV22*KSIZE*KSIZE+CONV22)
#define Layer31 (Layer22+ CONV22*CONV31*KSIZE*KSIZE+CONV31)
#define Layer32 (Layer31+ CONV31*CONV32*KSIZE*KSIZE+CONV32)
#define Layer33 (Layer32+ CONV32*CONV33*KSIZE*KSIZE+CONV33)
#define Layer41 (Layer33+ CONV33*CONV41*KSIZE*KSIZE+CONV41)
#define Layer42 (Layer41+ CONV41*CONV42*KSIZE*KSIZE+CONV42)
#define Layer43 (Layer42+ CONV42*CONV43*KSIZE*KSIZE+CONV43)
#define Layer51 (Layer43+ CONV43*CONV51*KSIZE*KSIZE+CONV51)
#define Layer52 (Layer51+ CONV51*CONV52*KSIZE*KSIZE+CONV52)
#define Layer53 (Layer52+ CONV52*CONV53*KSIZE*KSIZE+CONV53) 
#define WEIGHTSIZE Layer53

//#define INFM (DATA*CONV11_R*CONV11_R)
//add padding zeroes to deal with data re-order --only for input image;
#define INFM (UNROLL*CONV11_R*CONV11_R)
#define FM11 (CONV11*CONV11_R*CONV11_R)
#define FM12 (CONV12*CONV12_R*CONV12_R)
#define PL12 (POOL12*POOL12_R*POOL12_R)
#define FM21 (CONV21*CONV21_R*CONV21_R)
#define FM22 (CONV22*CONV22_R*CONV22_R)
#define PL22 (POOL22*POOL22_R*POOL22_R)
#define FM31 (CONV31*CONV31_R*CONV31_R)
#define FM32 (CONV32*CONV32_R*CONV32_R)
#define FM33 (CONV33*CONV33_R*CONV33_R)
#define PL33 (POOL33*POOL33_R*POOL33_R)
#define FM41 (CONV41*CONV41_R*CONV41_R)
#define FM42 (CONV42*CONV42_R*CONV42_R)
#define FM43 (CONV43*CONV43_R*CONV43_R)
#define PL43 (POOL43*POOL43_R*POOL43_R)
#define FM51 (CONV51*CONV51_R*CONV51_R)
#define FM52 (CONV52*CONV52_R*CONV52_R)
#define FM53 (CONV53*CONV53_R*CONV53_R)
#define PL53 (POOL53*POOL53_R*POOL53_R)

#if EXTN
} //extern "C"
#endif
