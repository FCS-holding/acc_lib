//#include <ap_int.h>
//#include <ap_fixed.h>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "ap_shift_reg.h"

//HWFIn_, hardware resource for input featuremap
//HWFOut_, hardware resource for output featuremap
//HWFR_, hardware resource for output featuremap's row dimension
//HWFC_, hardware resource for output featuremap's col dimension
//HWFinR_, hardware resource for input featuremap's row dimension
//HWFinC_, hardware resource for input featuremap's col dimension
//HWKsize_, hardware resource for kernel size
//HWin_, hardware unroll param for input
//HWout_, hardware unroll param for output

//fout, real number of output feature maps for a certain layer;
//fin, real number of input feature maps for a certain layer;
//frow, real output featuremap row for a certain layer;
//fcol, real output featuremap col for a certain layer;
//Ksize, real kernel size for a cerntain layer;
//Kstride, real kernel stride for a certain layer;
#include <iostream> 
#include <fstream> 
#include <string> 
#include "cnn_cfg.hpp" 
using namespace std;


template<typename data_t, int HWFOut_, int HWFR_, int HWFC_, int HWout_>
void ReluKernel2(data_t Pout[HWFOut_][HWFR_][HWFC_], ap_uint<16> frow, ap_uint<16> fcol) {
#pragma HLS inline off
#pragma HLS ARRAY_PARTITION variable=Pout complete dim=1
    ap_uint<16> i=0;
    ap_uint<16> j=0;
    data_t tmp[HWFOut_];
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=1

    for(ap_uint<16> l=0; l<frow*fcol; l++) {
#pragma HLS pipeline
#pragma HLS DEPENDENCE variable=Pout inter false
            for(ap_uint<16> k=0; k<HWFOut_; k++) {
#pragma HLS UNROLL
                tmp[k] = Pout[k][i][j];
            }
            for(ap_uint<16> k=0; k<HWFOut_; k++) {
#pragma HLS UNROLL
                if(tmp[k]<(data_t)0)  
                    Pout[k][i][j] = (data_t) 0;
            }
            j+=1;
            if(j>=fcol) {
                j=0;
                i+=1;
            }
    }
}

template<typename data_t, int HWFOut_, int HWFR_, int HWFC_, int HWout_>
void PoolKernel2(data_t Cout[HWFOut_][HWFR_][HWFC_], ap_uint<16> frow, ap_uint<16> fcol, ap_uint<16> kstride, ap_uint<16> ksize) {
#pragma HLS inline off
#pragma HLS ARRAY_PARTITION variable=Cout complete dim=1

data_t dmax[HWout_];
#pragma HLS ARRAY_PARTITION variable=dmax complete dim=1

    ap_uint<16> i=0;
    ap_uint<16> j=0;
    ap_uint<16> ktripcount = ksize*ksize;
    for(ap_uint<16> l=0; l<frow*fcol; l++) {
            for(ap_uint<16> k=0; k<HWFOut_; k++) {
#pragma HLS UNROLL 
                dmax[k] = Cout[k][kstride*i][kstride*j];
            } 
            ap_uint<16> c=0;
            ap_uint<16> r=0;
            for(ap_uint<16> p=0; p<ktripcount; p++) {
#pragma HLS pipeline 
                    for(ap_uint<16> k=0; k<HWFOut_; k++) {
#pragma HLS UNROLL 
                        if(Cout[k][kstride*i+r][kstride*j+c]>dmax[k]) {
                            dmax[k] = Cout[k][kstride*i+r][kstride*j+c];
                        }
                    }
                c+=1;
                if(c>=ksize) {
                    c=0;
                    r+=1;
                }
            }
            for(ap_uint<16> k=0; k<HWFOut_; k++) {
#pragma HLS UNROLL 
                Cout[k][i][j] = dmax[k];
            } 
        j+=1;
        if(j>=fcol) {
            j=0;
            i+=1;
        }
    }
}

template <typename data_t, typename bw_t, int HWFIn_, int HWFinR_, int HWFinC_, int HWFR_, int HWFC_>
void load_in(bool exec, data_t in[HWFIn_][HWFinR_][HWFinC_], bw_t *m, ap_uint<16> fin, ap_uint<16> ii, ap_uint<16> rr, ap_uint<16> HWin_, ap_uint<16> frow, ap_uint<16> fcol, ap_uint<16> finrow, ap_uint<16> fincol, ap_uint<16> prow, ap_uint<16> pad, int in_addr) {
#pragma HLS inline off
bw_t tmp;
int reorder_data_chunk_addr = ((int)ii*(int)frow*(int)fcol + in_addr)/HWFIn_ - (int)(pad*fcol) - (int)pad; 

if(exec) {
    for(ap_uint<16> tr=0; tr<prow+2*pad; tr++){
		ap_uint<16> trr = tr+rr;
        for(ap_uint<16> tc=0; tc<fincol; tc++){
#pragma HLS DEPENDENCE variable=in inter false
#pragma HLS pipeline

            tmp = m[reorder_data_chunk_addr + trr*fcol+ tc];
            if(!((trr>=pad)&&(tc>=pad)&&(trr<finrow-pad)&&(tc<fincol-pad))) {
                tmp = (bw_t)0;
            }
            for(ap_uint<16> ti=0; ti<HWin_; ti++) {
                ap_uint<DSIZE> idata = tmp.range(DSIZE*(ti+1)-1, DSIZE*ti);
                data_t *fdata = (data_t*)&idata;
                in[ti][tr][tc] = *fdata;
            }
        }
    }
}
}
template<typename data_t, int HWFOut_, int HWFIn_>
void shift_insert(data_t wdata[HWFIn_], data_t wt[HWFOut_][HWFIn_]){
#pragma HLS inline
    //manually implementing shifting register
    //shift each element downward(8->0) by 1 step and insert wdata to top
    for(int i=0; i<HWFOut_-1; i++) {
        for(int j=0; j<HWFIn_; j++) {
            wt[i][j] = wt[i+1][j]; 
        }
    }
    for(int j=0; j<HWFIn_; j++) {
        wt[HWFOut_-1][j] = wdata[j]; 
    }
    //end of shifting register
}

template <typename data_t, typename bw_t, int HWFIn_, int HWFOut_, int HWKsize_>
void load_weight(bool exec, data_t weight[HWFOut_][HWFIn_][HWKsize_][HWKsize_], bw_t* m, ap_uint<16> ii, ap_uint<16> oo, ap_uint<16> HWout_, ap_uint<16> HWin_, ap_uint<16> fin, ap_uint<16> Ksize, int weight_addr) {
#pragma HLS inline off
static data_t wt[HWFOut_][HWFIn_];
#pragma HLS ARRAY_PARTITION variable=wt complete dim=1
#pragma HLS ARRAY_PARTITION variable=wt complete dim=2
data_t wdata[HWFIn_];
#pragma HLS ARRAY_PARTITION variable=wdata complete dim=1
//int reorder_data_chunk_addr = (HWFOut_*HWFIn_*HWKsize_*HWKsize_)*(oo/HWFOut_*fin/HWFIn_ + ii/HWFIn_)/HWFIn_;
ap_uint<16> reorder_data_chunk_cont = Ksize*Ksize;
ap_uint<16> reorder_data_chunk_num  = ((fin<(ap_uint<16>)HWFIn_)?(ap_uint<16>)HWFIn_:fin)/(ap_uint<16>)HWFIn_;
int reorder_data_chunk_addr = (int)reorder_data_chunk_cont*((int)oo*(int)reorder_data_chunk_num + (int)ii) + weight_addr/HWFOut_;
    
if(exec) {
    ap_uint<16> tkr=0;
    ap_uint<16> tkc=0;
    ap_uint<16> to=0;
    loadweight:
    for(ap_uint<16> l=0; l<Ksize*Ksize*HWFOut_; l++) {
#pragma HLS DEPENDENCE variable=weight inter false
#pragma HLS pipeline
                bw_t tmp = m[ reorder_data_chunk_addr + (int)l];
                for(ap_uint<16> ti=0; ti<HWFIn_; ti++) {
                    ap_uint<DSIZE> idata = tmp.range((ti+1)*DSIZE-1, ti*DSIZE); 
                    data_t *fdata = (data_t*)&idata;
                    wdata[ti] = *fdata;
                }
                //manually implementing shifting register
                for(ap_uint<16> i=0; i<HWFOut_-1; i++) {
                    for(ap_uint<16> j=0; j<HWFIn_; j++) {
                        wt[i][j] = wt[i+1][j]; 
                    }
                }
                for(ap_uint<16> j=0; j<HWFIn_; j++) {
                    wt[HWFOut_-1][j] = wdata[j]; 
                }
                //end of shifting register
                for(ap_uint<16> too=0; too<HWFOut_; too++) {
#pragma HLS UNROLL
                    for(ap_uint<16> ti=0; ti<HWFIn_; ti++) {
#pragma HLS UNROLL
                            weight[too][ti][tkr][tkc] = wt[too][ti];
                    }
                }
        to+=1;
        if(to>=HWFOut_) {
            to=0;
            tkc+=1;
        }
        if(tkc>=Ksize) {
            tkc=0;
            tkr+=1;
        }
    }

}
}

template <typename data_t, typename bw_t, int HWFOut_, int HWFR_, int HWFC_>
void offload_output(bool exec, data_t Cout[HWFOut_][HWFR_][HWFC_], bw_t* m, ap_uint<16> fout, ap_uint<16> oo, ap_uint<16> rr, ap_uint<16> HWout_, ap_uint<16> frow, ap_uint<16> fcol, ap_uint<16> prow, int out_addr) {
#pragma HLS inline off
bw_t tmp;
int reorder_data_chunk_addr = (int)((int)oo*(int)fcol*(int)frow/HWFOut_) + (int)(rr*fcol) + out_addr/HWFOut_;

if(exec) {
    ap_uint<16> oj=0;
    ap_uint<16> ok=0;
    for(ap_uint<16> l=0; l<prow*fcol; l++) {
#pragma HLS DEPENDENCE variable=Cout inter false
#pragma HLS pipeline
            for(ap_uint<16> oi=0; oi<HWout_; oi++) {
                data_t fdata = Cout[oi][oj][ok];
                ap_uint<DSIZE> *idata = (ap_uint<DSIZE>*)&fdata;
                tmp.range((oi+1)*DSIZE-1, oi*DSIZE) = *idata;
            }
            m[ reorder_data_chunk_addr + (int)l ] = tmp;
        ok+=1;
        if(ok>=fcol) {
            ok=0;
            oj+=1;
        }
    }
}
}

template<typename data_t, typename bw_t, int HWFOut_, int HWFR_, int HWFC_, int HWout_>
void offload_output2(bool exec, data_t Cout[HWFOut_][HWFR_][HWFC_], bw_t *m, int out_addr, ap_uint<16> fout, ap_uint<16> rr, ap_uint<16> oo, ap_uint<16> frow, ap_uint<16> fcol, ap_uint<16> prow, int flag_pool, int flag_relu) {
#pragma HLS inline off
#pragma HLS ARRAY_PARTITION variable=Cout complete dim=1
    ap_uint<16> mprow, mcol, mrow;
    ap_uint<16> mrr;
	if(exec){
	    if(flag_pool) { 
	        mprow = prow/2;
	        mrow = frow/2;
	        mcol = fcol/2;
	        mrr = rr/2;
	        PoolKernel2<data_t, HWFOut_, HWFR_, HWFC_, HWout_>(Cout, mprow, mcol, 2, 2);
	    }
	    else {
	        mprow = prow;
	        mrow = frow;
	        mcol = fcol;
	        mrr = rr;
	    }
	    if(flag_relu) ReluKernel2<data_t, HWFOut_, HWFR_, HWFC_, HWout_>(Cout, mprow, mcol);
	    offload_output<data_t, bw_t, HWFOut_, HWFR_, HWFC_>(1, Cout, m, fout, oo, mrr, HWout_, mrow, mcol, mprow, out_addr);
	}
}

template<typename data_t, int HWFIn_, int HWFOut_, int HWFR_, int HWFC_, int HWFinR_, int HWFinC_, int HWKsize_, int HWin_, int HWout_>
void ConvCore(bool exec, data_t in[HWFIn_][HWFinR_][HWFinC_], data_t Cout[HWFOut_][HWFR_][HWFC_], data_t weight[HWFOut_][HWFIn_][HWKsize_][HWKsize_], data_t bias[HWFOut_], ap_uint<16> oo, ap_uint<16> ii, ap_uint<16> frow, ap_uint<16> fcol, ap_uint<16> prow, ap_uint<16> Ksize, ap_uint<16> Kstride, ap_uint<16> mask) {
#pragma HLS inline off
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=Cout complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=bias cyclic factor=HWout_ dim=1

data_t Ctmp[HWout_];
#pragma HLS ARRAY_PARTITION variable=Ctmp complete dim=1

if(exec){

    ap_uint<16> c=0;
    ap_uint<16> r=0;
    ap_uint<8> kc=0;
    ap_uint<8> kr=0;

#pragma HLS RESOURCE variable=Ctmp core=HAddSub_meddsp

    data_t tmp[HWin_];
    data_t tmp1[HWout_][HWin_];
    data_t tmp1_2[HWout_][HWin_];
    for(int l=0; l<Ksize*Ksize*prow*fcol; l++) {
#pragma HLS DEPENDENCE variable=Cout inter false
#pragma HLS pipeline
                    for(ap_uint<8> o=0; o<HWout_; o++){
#pragma HLS UNROLL
			            if((kr==0)&&(kc==0)&&(ii==0))   
                            Ctmp[o] = (data_t)bias[o+HWout_*(oo/HWout_)]; 
			            else	
                            Ctmp[o] = Cout[o][r][c];
		            }
                    for(ap_uint<8> i=0; i<HWin_; i++){
#pragma HLS UNROLL 
                        tmp[i] = in[i][r*Kstride+kr][c*Kstride+kc]; 
                        for(ap_uint<8> o=0; o<HWout_; o++){
#pragma HLS UNROLL
                            //data_t tmp1[o][i] = in[i][r*Kstride+kr][c*Kstride+kc] * weight[o][i][kr][kc]; 
                            tmp1[o][i] = tmp[i] * weight[o][i][kr][kc]; 
                            tmp1_2[o][i] = (i<mask)?tmp1[o][i]:(data_t)0; 
                            Ctmp[o] += tmp1_2[o][i];
                        }
                    }
                    for(ap_uint<8> o=0; o<HWout_; o++){
#pragma HLS UNROLL
		    	        Cout[o][r][c] = Ctmp[o];
		            }
        c+=1;
        if(c>=fcol) {
            c=0;
            r+=1;
        }
        if(r>=prow) {
            r=0;
            kc+=1;
        }
        if(kc>=Ksize) {
            kc=0;
            kr+=1;
        }
    }
}
}

template <typename data_t, typename bw_t, int HWFIn_, int HWFOut_, int HWFR_, int HWFC_, int HWFinR_, int HWFinC_, int HWKsize_, int HWin_, int HWout_>
void ConvKernel2(data_t in[HWFIn_][HWFinR_][HWFinC_], data_t Cout[HWFOut_][HWFR_][HWFC_], data_t weight[HWFOut_][HWFIn_][HWKsize_][HWKsize_], data_t in_1[HWFIn_][HWFinR_][HWFinC_], data_t weight_1[HWFOut_][HWFIn_][HWKsize_][HWKsize_], data_t bias[HWFOut_], int fin, int fout, int frow, int fcol, int finrow, int fincol, int Ksize, int Kstride, int pad, int mask, bw_t *m_fm, int in_addr, int weight_addr, int out_addr, int flag_pool, int flag_relu) {
#pragma HLS ARRAY_PARTITION variable=in complete dim=1
#pragma HLS ARRAY_PARTITION variable=Cout complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=bias cyclic factor=HWout_ dim=1

    //---conv---//
    ap_uint<16> i, o, r, c; // i, o, r, c are for input(HWFIn_), output (HWFOut_), row(HWFR_), col(HWFC_)
    ap_uint<16> oo, ii, rr;
    ap_uint<16> kr, kc;  //kr, kc for kernel loops (Ksize)
    ap_uint<16> tr, tc, ti;
    ap_uint<16> to, tkr, tkc;
    bool pinpon_flag_in=0;
    bool pinpon_flag_out=0;
    ap_uint<16> prow; //--try: tile row: row loop factor after partition--//
    ap_uint<16> mrow, mprow, mcol, mrr; // for memory copy (offload_output) operation parameters

    int bias_addr = fout*((fin<HWFIn_)?HWFIn_:fin)*Ksize*Ksize/HWFIn_;
    load_bias:
    for(int bi=0; bi<fout/HWFIn_; bi++) {
        bw_t btmp = m_fm[bi + bias_addr + weight_addr/HWFIn_];
        for(int bj=0; bj<HWFIn_; bj++) {
#pragma HLS pipeline
            ap_uint<DSIZE> ibtmp = btmp.range((bj+1)*DSIZE-1, DSIZE*bj);
            data_t *fbtmp = (data_t*)&ibtmp;
            bias[bi*HWFIn_+ bj] = *fbtmp; 
        }
    }
    shell_loop:
    for(oo=0; oo<fout; oo=oo+HWout_){
        for(rr=0; rr<frow; rr=rr+HWFR_) { //--try: tile row--//
            ap_uint<16> prow_factor = frow-rr;
            prow = (prow_factor>=(ap_uint<16>)HWFR_)?(ap_uint<16>)HWFR_:prow_factor;
        	    for(ii=0; ii<fin+HWin_; ii=ii+HWin_){
                    bool flag_in = (ii<fin)?1:0;
                    bool flag_con = (ii>0)?1:0;
        	        if(pinpon_flag_in == 1) {
        	            load_in<data_t, bw_t, HWFIn_, HWFinR_, HWFinC_, HWFR_, HWFC_>((flag_in), in, m_fm, fin, ii, rr, HWin_, frow, fcol, finrow, fincol, prow, pad, in_addr); 
        	            load_weight<data_t, bw_t, HWFIn_, HWFOut_, HWKsize_>((flag_in), weight, m_fm, ii, oo, HWout_, HWin_, fin, Ksize, weight_addr);
        	        
        	            ConvCore<data_t, HWFIn_, HWFOut_, HWFR_, HWFC_, HWFinR_, HWFinC_, HWKsize_, HWin_, HWout_>
        	            ((flag_con), in_1, Cout, weight_1, bias, oo, (ii-HWin_), frow, fcol, prow, Ksize, Kstride, mask); 
        	        }
        	        else {
        	            load_in<data_t, bw_t, HWFIn_, HWFinR_, HWFinC_, HWFR_, HWFC_>((flag_in), in_1, m_fm, fin, ii, rr, HWin_, frow, fcol, finrow, fincol, prow, pad, in_addr); 
        	            load_weight<data_t, bw_t, HWFIn_, HWFOut_, HWKsize_>((flag_in), weight_1, m_fm, ii, oo, HWout_, HWin_, fin, Ksize, weight_addr);
        	        
        	            ConvCore<data_t, HWFIn_, HWFOut_, HWFR_, HWFC_, HWFinR_, HWFinC_, HWKsize_, HWin_, HWout_>
        	            ((flag_con), in, Cout, weight, bias, oo, (ii-HWin_), frow, fcol, prow, Ksize, Kstride, mask); 
        	        }
        	        pinpon_flag_in = 1 - pinpon_flag_in;
        	    }
                offload_output2<data_t, bw_t, HWFOut_, HWFR_, HWFC_, HWout_>(1, Cout, m_fm, out_addr, fout, rr, oo, frow, fcol, prow, flag_pool, flag_relu); 
        } //--try: tile row--//
        //printf("\n");
    }
}


