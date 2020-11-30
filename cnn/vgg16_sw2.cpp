//#include "cnn_cfg.hpp"
#include "vgg16.hpp"

template<typename data_t, int FIN, int FOUT, int Kernel>
void init_weights(data_t *m, int addr, data_t *weights, data_t *bias) {
    for(int i=0; i<FOUT; i++){
        bias[i] = m[addr+ FIN*FOUT*Kernel*Kernel + i];
        for(int j=0; j<FIN; j++){
            for(int k=0; k<Kernel; k++) {
                for(int s=0; s<Kernel; s++) {
                    weights[i*FIN*Kernel*Kernel + j*Kernel*Kernel + k*Kernel + s] = m[addr + i*FIN*Kernel*Kernel + j*Kernel*Kernel + k*Kernel + s];
    }   }   }   }
}

template<typename data_t, int FIN_, int FOUT_, int FINR_, int FINC_, int FOUTR_, int FOUTC_, int KSIZE_, int KSTRIDE_>
void conv_sw(data_t *in, data_t *Cout, data_t *weight, data_t *bias)
{
    //---conv---//
    int Couti, Coutj, Coutk;
    int Cini;
    int Cwi, Cwj;

    for(Coutj=0; Coutj<FOUTR_; Coutj++){
        for(Coutk=0; Coutk<FOUTC_; Coutk++){
            for(Couti=0; Couti<FOUT_; Couti++){
                Cout[Couti*FOUTR_*FOUTC_ +Coutj*FOUTC_ +Coutk] = (data_t)bias[Couti];
                for(Cini=0; Cini<FIN_; Cini++){
                    for(Cwi=0; Cwi<KSIZE_; Cwi++){
                        for(Cwj=0; Cwj<KSIZE_; Cwj++){
                            Cout[Couti*FOUTR_*FOUTC_ +Coutj*FOUTC_ +Coutk] +=
                            in[Cini*FINR_*FINC_ +(Coutj*KSTRIDE_+Cwi)*FINC_ +(Coutk*KSTRIDE_+Cwj)] * weight[Couti*FIN_*KSIZE_*KSIZE_ +Cini*KSIZE_*KSIZE_+ Cwi*KSIZE_ + Cwj];
    }   }   }   }   }   }
}
template<typename data_t, int FOUT_, int FOUTR_, int FOUTC_>
void relu_sw(data_t *Pout)
{
    //---relu---//
    for(int i=0; i<FOUT_*FOUTR_*FOUTC_; i++){
        if(Pout[i] < (data_t)0)
            Pout[i] = (data_t)0;
    }
}
template<typename data_t, int FOUT_, int FINR_, int FINC_, int FOUTR_, int FOUTC_, int KSIZE_, int KSTRIDE_>
void pool_sw( data_t *Cout, data_t *Pout)
{
    //---pool---//
    int Pouti, Poutj, Poutk;
    int Pini, Pinj;
    for(Pouti=0; Pouti<FOUT_; Pouti++){
        for(Poutj=0; Poutj<FOUTR_; Poutj++){
            for(Poutk=0; Poutk<FOUTC_; Poutk++){
                Pout[Pouti*FOUTR_*FOUTC_ + Poutj*FOUTC_ + Poutk] = Cout[Pouti*FINR_*FINC_ + Poutj*KSTRIDE_*FINC_ + Poutk*KSTRIDE_];
                for(Pini=0; Pini<KSIZE_; Pini++){
                    for(Pinj=0; Pinj<KSIZE_; Pinj++){
                        if(Cout[Pouti*FINC_*FINR_+ (Poutj*KSTRIDE_+Pini)*FINC_ + (Poutk*KSTRIDE_+Pinj)] > Pout[Pouti*FOUTR_*FOUTC_ + Poutj*FOUTC_ + Poutk])
                            Pout[Pouti*FOUTR_*FOUTC_ + Poutj*FOUTC_ + Poutk] = Cout[Pouti*FINC_*FINR_+ (Poutj*KSTRIDE_+Pini)*FINC_ + (Poutk*KSTRIDE_+Pinj)];
     }   }   }   }   }
}
template<typename data_t, int FOUT, int FR, int FC>
void dram_op(data_t *m, int addr, data_t *in, int pad, int F)
{
    for(int i=0; i<FOUT; i++) {
        for(int j=0; j<FR; j++) {
            for(int k=0; k<FC; k++) {
                if(F==READ) {
                    if((k>=pad)&&(j>=pad)&&(k<FC-pad)&&(j<FR-pad))
                        in[i*FR*FC+ j*FC+ k] = m[addr+ i*(FR-2*pad)*(FC-2*pad)+ (j-pad)*(FC-2*pad) + (k-pad)];
                    else in[i*FR*FC+ j*FC+ k] = (data_t) 0;
                }
                else { 
                    m[addr+ i*FR*FC+ j*FC+ k] = in[i*FR*FC+ j*FC+ k];
                }
            }
        }
    }
}
template<typename data_t, int FOUT_, int FINR_, int FINC_, int FOUTR_, int FOUTC_>
void transfer(data_t *in, data_t *Cout)
{
    for(int i=0; i<FOUT_; i++) {
        for(int j=0; j<FINR_; j++) {
            for(int k=0; k<FINC_; k++) {
                if((j>=1)&&(k>=1)&&(j<FINR_-1)&&(k<FINC_-1))
                    in[i*FINR_*FINC_ +j*FINC_ +k] = Cout[i*FOUTR_*FOUTC_+ (j-1)*FOUTC_+ (k-1)];
                else
                    in[i*FINR_*FINC_ +j*FINC_ +k] = (data_t) 0;
    }   }   }
}

void vgg16_sw2(data_t *m) {

static data_t Cin[226*226*64];
static data_t Cout[226*226*64];
static data_t Pout[226*226*64];
static data_t Weight[512*512*3*3];
static data_t bias[512];

dram_op<data_t, DATA, CONV11_I, CONV11_I>(m, WEIGHTSIZE, Cin, PAD, READ);
//---layer1_1----
init_weights<data_t, DATA, CONV11, KSIZE>(m, 0, Weight, bias);
conv_sw<data_t, DATA, CONV11, CONV11_I, CONV11_I, CONV11_R, CONV11_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV11, CONV11_R, CONV11_R>(Cout);
//dram_op<data_t, CONV11, CONV11_R, CONV11_R>(m, WEIGHTSIZE+ INFM , Cout, PAD, WRITE);

//---layer1_2----
transfer<data_t, CONV11, CONV12_I, CONV12_I, CONV11_R, CONV11_R>(Cin, Cout);
init_weights<data_t, CONV11, CONV12, KSIZE>(m, Layer11, Weight, bias);
conv_sw<data_t, CONV11, CONV12, CONV12_I, CONV12_I, CONV12_R, CONV12_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV12, CONV12_R, CONV12_R>(Cout);
//dram_op<data_t, CONV12, CONV12_R, CONV12_R>(m, WEIGHTSIZE+ INFM , Cout, PAD, WRITE);
pool_sw<data_t, POOL12, CONV12_R, CONV12_R, POOL12_R, POOL12_R, 2, 2>(Cout, Pout);
//dram_op<data_t, POOL12, POOL12_R, POOL12_R>(m, (WEIGHTSIZE+ INFM), Pout, PAD, WRITE);

//---layer2_1----
transfer<data_t, POOL12, CONV21_I, CONV21_I, POOL12_R, POOL12_R>(Cin, Pout);
init_weights<data_t, CONV12, CONV21, KSIZE>(m, Layer12, Weight, bias);
conv_sw<data_t, POOL12, CONV21, CONV21_I, CONV21_I, CONV21_R, CONV21_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV21, CONV21_R, CONV21_R>(Cout);
//dram_op<data_t, CONV21, CONV21_R, CONV21_R>(m, (WEIGHTSIZE+ INFM), Cout, PAD, WRITE);
//---layer2_2----
transfer<data_t, CONV21, CONV22_I, CONV22_I, CONV21_R, CONV21_R>(Cin, Cout);
init_weights<data_t, CONV21, CONV22, KSIZE>(m, Layer21, Weight, bias);
conv_sw<data_t, CONV21, CONV22, CONV22_I, CONV22_I, CONV22_R, CONV22_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV22, CONV22_R, CONV22_R>(Cout);
pool_sw<data_t, POOL22, CONV22_R, CONV22_R, POOL22_R, POOL22_R, 2, 2>(Cout, Pout);
//dram_op<data_t, POOL22, POOL22_R, POOL22_R>(m, (WEIGHTSIZE+ INFM), Pout, PAD, WRITE);

//---layer3_1----
transfer<data_t, POOL22, CONV31_I, CONV31_I, POOL22_R, POOL22_R>(Cin, Pout);
init_weights<data_t, CONV22, CONV31, KSIZE>(m, Layer22, Weight, bias);
conv_sw<data_t, POOL22, CONV31, CONV31_I, CONV31_I, CONV31_R, CONV31_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV31, CONV31_R, CONV31_R>(Cout);
//dram_op<data_t, CONV31, CONV31_R, CONV31_R>(m, (WEIGHTSIZE+ DATA*CONV11_R*CONV11_R), Cout31, PAD, WRITE);
//---layer3_2----
transfer<data_t, CONV31, CONV32_I, CONV32_I, CONV31_R, CONV31_R>(Cin, Cout);
init_weights<data_t, CONV31, CONV32, KSIZE>(m, Layer31, Weight, bias);
conv_sw<data_t, CONV31, CONV32, CONV32_I, CONV32_I, CONV32_R, CONV32_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV32, CONV32_R, CONV32_R>(Cout);
//dram_op<data_t, CONV32, CONV32_R, CONV32_R>(m, (WEIGHTSIZE+ DATA*CONV11_R*CONV11_R), Cout32, PAD, WRITE);
//--- layer3_3----
transfer<data_t, CONV32, CONV33_I, CONV33_I, CONV32_R, CONV32_R>(Cin, Cout);
init_weights<data_t, CONV32, CONV33, KSIZE>(m, Layer32, Weight, bias);
conv_sw<data_t, CONV32, CONV33, CONV33_I, CONV33_I, CONV33_R, CONV33_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV33, CONV33_R, CONV33_R>(Cout);
pool_sw<data_t, POOL33, CONV33_R, CONV33_R, POOL33_R, POOL33_R, 2, 2>(Cout, Pout);
dram_op<data_t, POOL33, POOL33_R, POOL33_R>(m, (WEIGHTSIZE+ INFM), Pout, PAD, WRITE);

//---layer4_1----
transfer<data_t, POOL33, CONV41_I, CONV41_I, POOL33_R, POOL33_R>(Cin, Pout);
init_weights<data_t, CONV33, CONV41, KSIZE>(m, Layer33, Weight, bias);
conv_sw<data_t, POOL33, CONV41, CONV41_I, CONV41_I, CONV41_R, CONV41_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV41, CONV41_R, CONV41_R>(Cout);
//dram_op<data_t, CONV41, CONV41_R, CONV41_R>(m, (WEIGHTSIZE+ DATA*CONV11_R*CONV11_R), Cout41, PAD, WRITE);
//---layer4_2----
transfer<data_t, CONV41, CONV42_I, CONV42_I, CONV41_R, CONV41_R>(Cin, Cout);
init_weights<data_t, CONV41, CONV42, KSIZE>(m, Layer41, Weight, bias);
conv_sw<data_t, CONV41, CONV42, CONV42_I, CONV42_I, CONV42_R, CONV42_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV42, CONV42_R, CONV42_R>(Cout);
//dram_op<data_t, CONV42, CONV42_R, CONV42_R>(m, (WEIGHTSIZE+ DATA*CONV11_R*CONV11_R), Cout42, PAD, WRITE);
//--- layer4_3----
transfer<data_t, CONV42, CONV43_I, CONV43_I, CONV42_R, CONV42_R>(Cin, Cout);
init_weights<data_t, CONV42, CONV43, KSIZE>(m, Layer42, Weight, bias);
conv_sw<data_t, CONV42, CONV43, CONV43_I, CONV43_I, CONV43_R, CONV43_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV43, CONV43_R, CONV43_R>(Cout);
pool_sw<data_t, POOL43, CONV43_R, CONV43_R, POOL43_R, POOL43_R, 2, 2>(Cout, Pout);
//dram_op<data_t, POOL43, POOL43_R, POOL43_R>(m, (WEIGHTSIZE+ INFM), Pout, PAD, WRITE);

//---layer5_1----
transfer<data_t, POOL43, CONV51_I, CONV51_I, POOL43_R, POOL43_R>(Cin, Pout);
init_weights<data_t, CONV43, CONV51, KSIZE>(m, Layer43, Weight, bias);
conv_sw<data_t, POOL43, CONV51, CONV51_I, CONV51_I, CONV51_R, CONV51_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV51, CONV51_R, CONV51_R>(Cout);
//dram_op<data_t, CONV51, CONV51_R, CONV51_R>(m, (WEIGHTSIZE+ DATA*CONV11_R*CONV11_R), Cout51, PAD, WRITE);
//---layer5_2----
transfer<data_t, CONV51, CONV52_I, CONV52_I, CONV51_R, CONV51_R>(Cin, Cout);
init_weights<data_t, CONV51, CONV52, KSIZE>(m, Layer51, Weight, bias);
conv_sw<data_t, CONV51, CONV52, CONV52_I, CONV52_I, CONV52_R, CONV52_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV52, CONV52_R, CONV52_R>(Cout);
//dram_op<data_t, CONV52, CONV52_R, CONV52_R>(m, (WEIGHTSIZE+ DATA*CONV11_R*CONV11_R), Cout52, PAD, WRITE);
//--- layer5_3----
transfer<data_t, CONV52, CONV53_I, CONV53_I, CONV52_R, CONV52_R>(Cin, Cout);
init_weights<data_t, CONV52, CONV53, KSIZE>(m, Layer52, Weight, bias);
conv_sw<data_t, CONV52, CONV53, CONV53_I, CONV53_I, CONV53_R, CONV53_R, KSIZE, STRIDE>(Cin, Cout, Weight, bias);
relu_sw<data_t, CONV53, CONV53_R, CONV53_R>(Cout);
pool_sw<data_t, POOL53, CONV53_R, CONV53_R, POOL53_R, POOL53_R, 2, 2>(Cout, Pout);
dram_op<data_t, POOL53, POOL53_R, POOL53_R>(m, (WEIGHTSIZE+ INFM), Pout, PAD, WRITE);

}



