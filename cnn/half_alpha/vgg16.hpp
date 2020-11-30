#include "cnn_cfg.hpp"

#if EXTN
extern "C" {
#endif

typedef struct lyr
{
    int fin;
    int fout;
    int finrow;
    int fincol;
    int frow;
    int fcol;
    int Ksize;
    int Kstride;
    int pad;
    int mask;
    int addr_in;
    int addr_wght;
    int addr_out;
    int pool;
    int relu;
} layer;

extern layer conv1_1 ;
extern layer conv1_2 ;
extern layer pool1_2 ;
extern layer conv2_1 ;
extern layer conv2_2 ;
extern layer pool2_2 ;
extern layer conv3_1 ;
extern layer conv3_2 ;
extern layer conv3_3 ;
extern layer pool3_3 ;
extern layer conv4_1 ;
extern layer conv4_2 ;
extern layer conv4_3 ;
extern layer pool4_3 ;
extern layer conv5_1 ;
extern layer conv5_2 ;
extern layer conv5_3 ;
extern layer pool5_3 ;

//extern int addr_data  ;
//extern int addr_out11 ;
//extern int addr_out12 ;
//extern int addr_pool12;
//extern int addr_out21 ;
//extern int addr_out22 ;
//extern int addr_pool22;
//extern int addr_out31 ;
//extern int addr_out32 ;
//extern int addr_out33 ;
//extern int addr_pool33;
//extern int addr_out41 ;
//extern int addr_out42 ;
//extern int addr_out43 ;
//extern int addr_pool43;
//extern int addr_out51 ;
//extern int addr_out52 ;
//extern int addr_out53 ;
//extern int addr_pool53;
//
//extern int addr_wght11 ;
//extern int addr_wght12 ;
//extern int addr_wght21 ;
//extern int addr_wght22 ;
//extern int addr_wght31 ;
//extern int addr_wght32 ;
//extern int addr_wght33 ;
//extern int addr_wght41 ;
//extern int addr_wght42 ;
//extern int addr_wght43 ;
//extern int addr_wght51 ;
//extern int addr_wght52 ;
//extern int addr_wght53 ;


void vgg16(bw_t* m_fm);
void vgg16_sw2(float *m);

#if EXTN
}
#endif

