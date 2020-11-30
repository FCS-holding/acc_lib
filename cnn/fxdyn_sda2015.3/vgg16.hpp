#include "cnn_cfg.hpp"
#include "falconML.hpp"

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

void vgg16(bw_t* m_fm, ly_t* ly_dram);
void vgg16_sw2(data_t *m);

//LayerDef+ to the following for DRAM space shift
int addr_data   = ( WEIGHTSIZE);
int addr_out11  = ( WEIGHTSIZE+ INFM);
int addr_out12  = ( WEIGHTSIZE+ INFM+ FM11);
int addr_pool12 = ( WEIGHTSIZE+ INFM+ FM11 + FM12);
int addr_out21  = ( WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12);
int addr_out22  = ( WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21);
int addr_pool22 = ( WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22);
int addr_out31  = ( WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22);
int addr_out32  = ( WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31);
int addr_out33  = ( WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32);
int addr_pool33 = ( WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32+ FM33);
int addr_out41  = ( WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32+ FM33+ PL33);
int addr_out42  = ( WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32+ FM33+ PL33+ FM41);
int addr_out43  = ( WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32+ FM33+ PL33+ FM41+ FM42);
int addr_pool43 = ( WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32+ FM33+ PL33+ FM41+ FM42+ FM43);
int addr_out51  = ( WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32+ FM33+ PL33+ FM41+ FM42+ FM43+ PL43);
int addr_out52  = ( WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32+ FM33+ PL33+ FM41+ FM42+ FM43+ PL43+ FM51);
int addr_out53  = ( WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32+ FM33+ PL33+ FM41+ FM42+ FM43+ PL43+ FM51+ FM52);
int addr_pool53 = ( WEIGHTSIZE+ INFM+ FM11+ FM12+ PL12+ FM21+ FM22+ PL22+ FM31+ FM32+ FM33+ PL33+ FM41+ FM42+ FM43+ PL43+ FM51+ FM52+ FM53);

//LayerDef+ to the following for DRAM space shift
int addr_wght11 = 0;
int addr_wght12 = Layer11;
int addr_wght21 = Layer12;
int addr_wght22 = Layer21;
int addr_wght31 = Layer22;
int addr_wght32 = Layer31;
int addr_wght33 = Layer32;
int addr_wght41 = Layer33;
int addr_wght42 = Layer41;
int addr_wght43 = Layer42;
int addr_wght51 = Layer43;
int addr_wght52 = Layer51;
int addr_wght53 = Layer52;


//-------------- in, out, ir, ic,  r,  c, k,  s, pad, mask ------- 
layer conv1_1 = {DATA, CONV11, CONV11_I, CONV11_I, CONV11_R, CONV11_R, KSIZE, STRIDE, PAD, DATA, addr_data, addr_wght11, addr_out11, 0, RELU};
layer conv1_2 = {CONV11, CONV12, CONV12_I, CONV12_I, CONV12_R, CONV12_R, KSIZE, STRIDE, PAD, HWin, addr_out11, addr_wght12, addr_pool12, POOL, RELU};
//layer pool1_2 = {CONV11, POOL12, CONV12_R, CONV12_R, POOL12_R, POOL12_R, 2, 2, 0, 0, addr_pool12, addr_wght21, addr_out21, 0, RELU};
layer conv2_1 = {POOL12, CONV21, CONV21_I, CONV21_I, CONV21_R, CONV21_R, KSIZE, STRIDE, PAD, HWin, addr_pool12, addr_wght21, addr_out21, 0, RELU};
layer conv2_2 = {CONV21, CONV22, CONV22_I, CONV22_I, CONV22_R, CONV22_R, KSIZE, STRIDE, PAD, HWin, addr_out21, addr_wght22, addr_pool22, POOL, RELU};
//layer pool2_2 = {CONV22, POOL22, CONV22_R, CONV22_R, POOL22_R, POOL22_R, 2, 2, 0, 0};
layer conv3_1 = {POOL22, CONV31, CONV31_I, CONV31_I, CONV31_R, CONV31_R, KSIZE, STRIDE, PAD, HWin, addr_pool22, addr_wght31, addr_out31, 0, RELU};
layer conv3_2 = {CONV31, CONV32, CONV32_I, CONV32_I, CONV32_R, CONV32_R, KSIZE, STRIDE, PAD, HWin, addr_out31, addr_wght32, addr_out32, 0, RELU};
layer conv3_3 = {CONV32, CONV33, CONV33_I, CONV33_I, CONV33_R, CONV33_R, KSIZE, STRIDE, PAD, HWin, addr_out32, addr_wght33, addr_pool33, POOL, RELU};
//layer pool3_3 = {CONV33, POOL33, CONV33_R, CONV33_R, POOL33_R, POOL33_R, 2, 2, 0, 0};
layer conv4_1 = {POOL33, CONV41, CONV41_I, CONV41_I, CONV41_R, CONV41_R, KSIZE, STRIDE, PAD, HWin, addr_pool33, addr_wght41, addr_out41, 0, RELU};
layer conv4_2 = {CONV41, CONV42, CONV42_I, CONV42_I, CONV42_R, CONV42_R, KSIZE, STRIDE, PAD, HWin, addr_out41, addr_wght42, addr_out42, 0, RELU};
layer conv4_3 = {CONV42, CONV43, CONV43_I, CONV43_I, CONV43_R, CONV43_R, KSIZE, STRIDE, PAD, HWin, addr_out42, addr_wght43, addr_pool43, POOL, RELU};
//layer pool4_3 = {CONV43, POOL43, CONV43_R, CONV43_R, POOL43_R, POOL43_R, 2, 2, 0, 0};
layer conv5_1 = {POOL43, CONV51, CONV51_I, CONV51_I, CONV51_R, CONV51_R, KSIZE, STRIDE, PAD, HWin, addr_pool43, addr_wght51, addr_out51, 0, RELU};
layer conv5_2 = {CONV51, CONV52, CONV52_I, CONV52_I, CONV52_R, CONV52_R, KSIZE, STRIDE, PAD, HWin, addr_out51, addr_wght52, addr_out52, 0, RELU};
layer conv5_3 = {CONV52, CONV53, CONV53_I, CONV53_I, CONV53_R, CONV53_R, KSIZE, STRIDE, PAD, HWin, addr_out52, addr_wght53, addr_pool53, POOL, RELU};
//layer pool5_3 = {CONV53, POOL53, CONV53_R, CONV53_R, POOL53_R, POOL53_R, 2, 2, 0, 0};


#if EXTN
}
#endif

