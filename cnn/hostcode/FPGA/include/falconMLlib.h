#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <stdexcept>

#include <caffe/caffe.hpp>
#include <CL/opencl.h>

#define SIM 0

using namespace caffe;
using std::vector;
using std::string;
using std::ofstream;
using std::ifstream;
using std::endl;
using std::cout;

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

typedef struct layer_config {
    string type;
    int num_input;
    int num_output;
    int input_height;
    int input_width;
    int output_height;
    int output_width;
    int kernel_size;
    int kernel_stride;
    int kernel_pad;
} lycfg;

float rcmp(float a, float b){
    return fabs((a-b)/(a+b));
}

class CNN4FPGA {
    public:
        //CNN4FPGA(){}

        int weight_len(){
            return wght_length ;
        }
        int layerdef_len(){
            return layerdef_length;
        }
        int fm_len(){
            return fm_length;
        }
        int infm_len(){
            return infm_length;
        }
        int lastfm_len(){
            return lastfm_length;
        }
        float* weight_ptr(){
            return wght_dram;
        }
        int* layerdef_ptr(){
            return ly_host;
        }
        ~CNN4FPGA() {
            free(wght_dram); 
            free(ly_host); 
        }
    public:
        void setCNNModel(shared_ptr< Net< float > > net_);
        vector< lycfg > origin;
        vector< layer > net4fpga;
    private:
        vector< lycfg > ParseCNN_Arch(shared_ptr<Net<float> > net_);
        vector< layer > EstabNet4FPGA(vector< lycfg > layer_cfg, int* wght_length, int* layerdef_length, int* fm_length, int* infm_length, int* lastfm_length);
        void PrepareLayerdef(vector< layer > b, int* ly_host);
        void PrepareWght(shared_ptr<Net<float> > net_, vector< lycfg > a, vector< layer > b, float* wght_dram);

    //for debug

        void print_lycfg(vector< lycfg > origin);
        void print_layer(vector< layer > net4fpga); 
        void verify_reorder(float* wght_dram);
    private:
        int layerdef_length;
        int wght_length;
        int fm_length;
        int infm_length;
        int lastfm_length;
        float* wght_dram; 
        int* ly_host;
        int DUM;

};


void CNN4FPGA::setCNNModel(shared_ptr<Net<float> > net_) {
    DUM = UNROLL;
    origin =  ParseCNN_Arch(net_); 
    print_lycfg(origin);

    net4fpga = EstabNet4FPGA(origin, &wght_length, &layerdef_length, &fm_length, &infm_length, &lastfm_length);
    std::cout <<" wght len: "<<wght_length<<" layer len: "<<layerdef_length<<" fm len: "<<fm_length<<" last fm len: "<<lastfm_length<<std::endl;
    //std::cout << "wght_length: " << wght_length << " layerdef_length: "<< layerdef_length << std::endl;
    print_layer(net4fpga);

    wght_dram = (float*)malloc(wght_length*sizeof(float)); 
    ly_host = (int*)malloc(layerdef_length*sizeof(int));
    memset(ly_host, 0, layerdef_length*sizeof(int));
    memset(wght_dram, 0, wght_length*sizeof(float));
    PrepareLayerdef(net4fpga, ly_host);
    PrepareWght(net_, origin, net4fpga, wght_dram);
    //verify_reorder(wght_dram);
    std::cout << "finish pareparing CNN model for FPGA accelerator" << std::endl;
}

void CNN4FPGA::PrepareWght(shared_ptr<Net<float> > net_, vector< lycfg > a, vector< layer > b, float* wght_dram){

  vector< lycfg > cast;
  for(int i=0; i<a.size(); i++){
    if(a[i].type.compare("conv")==0) {
        cast.push_back(a[i]);
    }
  }

  vector< std::string > layer_name = net_->layer_names();
  std::cout << layer_name.size() << std::endl;
  int layer_num = layer_name.size();
  int j=0;
  int addr = 0;

  //ofstream curr;
  //curr.open("layer1_curr.txt", ios::app);

  for(int i=0; i<layer_num; i++){
    if(layer_name[i].compare(0, 4, "conv")==0) {
        shared_ptr< Layer< float > > layer = net_->layers()[i];
        vector< shared_ptr< Blob< float > > > blobs = layer->blobs();
        const float * weight_ptr = (const float *) blobs[0]->cpu_data();
        const float * bias_ptr = (const float *) blobs[1]->cpu_data();
        //read weights out and add pading
        float* tmp_dram = (float*)malloc(b[j].fout*b[j].fin*b[j].Ksize*b[j].Ksize*sizeof(float)); 
        for(int u=0; u<b[j].fout; u++){
            for(int k=0; k<b[j].fin; k++){
                for(int h=0; h<b[j].Ksize*b[j].Ksize; h++){
                    if((u<cast[j].num_output)&&(k<cast[j].num_input)) {
                        tmp_dram[u*b[j].fin*b[j].Ksize*b[j].Ksize + k*b[j].Ksize*b[j].Ksize + h] =\
                                    weight_ptr[ u*cast[j].num_input*cast[j].kernel_size*cast[j].kernel_size + \
                                                k*cast[j].kernel_size*cast[j].kernel_size + h]; 

                    }
                    else {
                        tmp_dram[u*b[j].fin*b[j].Ksize*b[j].Ksize + k*b[j].Ksize*b[j].Ksize + h] = 0.0;
                    }
                }
            }
        }
        //reorder this layer's weight
        for(int uu=0; uu<b[j].fout; uu+=HWFOut){
            for(int yy=0; yy<b[j].fin; yy+=HWFIn){
                for(int k=0; k<b[j].Ksize*b[j].Ksize; k++){
                    for(int u=0; u<HWFOut; u++){
                        for(int y=0; y<HWFIn; y++){
                            wght_dram[addr+ uu*b[j].fin*b[j].Ksize*b[j].Ksize + \
                                            yy* b[j].Ksize*b[j].Ksize*HWFIn + k*HWFOut*HWFIn + u*HWFIn + y] = \
                                                tmp_dram[(uu+u)*b[j].fin*b[j].Ksize*b[j].Ksize+ (yy+y)*b[j].Ksize*b[j].Ksize+ k];
                            //if(j==0) {
                            //    curr<<(uu+u)<<"\t"<<(yy+y)<<"\t"<<k<<"\t"<<tmp_dram[(uu+u)*b[j].fin*b[j].Ksize*b[j].Ksize+ (yy+y)*b[j].Ksize*b[j].Ksize+ k]<< std::endl;
                            //}
                        }
                    }
                }
            }
        }
        for(int uu=0; uu<b[j].fout; uu++){
            if(uu<cast[j].num_output)
                wght_dram[addr+ b[j].fout*b[j].fin*b[j].Ksize*b[j].Ksize+ uu] = bias_ptr[uu]; 
            else
                wght_dram[addr+ b[j].fout*b[j].fin*b[j].Ksize*b[j].Ksize+ uu] = 0.0;
        }
        addr += (b[j].fout*b[j].fin*b[j].Ksize*b[j].Ksize + b[j].fout);
        free(tmp_dram);
        j++;
    }
  }
  //curr.close(); 
}

void CNN4FPGA::PrepareLayerdef(vector< layer > b, int* ly_host) {
    ly_host[14] = b.size(); 
    ly_host[15] = b.size(); 
    //std::cout << "fin\tfout\tfinrow\tfincol\tfrow\tfcol\tKsize\tKstride\tpad\tmask\taddr_in\taddr_wght\taddr_out\tpool\trelu\t0" << std::endl;
    for(int i=0; i<b.size(); i++){
        ly_host[16+ i*16+ 0] = b[i].fin;
        ly_host[16+ i*16+ 1] = b[i].fout;
        ly_host[16+ i*16+ 2] = b[i].finrow;
        ly_host[16+ i*16+ 3] = b[i].fincol;
        ly_host[16+ i*16+ 4] = b[i].frow;
        ly_host[16+ i*16+ 5] = b[i].fcol;
        ly_host[16+ i*16+ 6] = b[i].Ksize;
        ly_host[16+ i*16+ 7] = b[i].Kstride;
        ly_host[16+ i*16+ 8] = b[i].pad;
        ly_host[16+ i*16+ 9] = b[i].mask;
        ly_host[16+ i*16+ 10] = b[i].addr_in;
        ly_host[16+ i*16+ 11] = b[i].addr_wght;
        ly_host[16+ i*16+ 12] = b[i].addr_out;
        ly_host[16+ i*16+ 13] = b[i].pool;
        ly_host[16+ i*16+ 14] = b[i].relu;
        ly_host[16+ i*16+ 15] = 0; 
        //std::cout <<  b[i].fin << "\t" << b[i].fout << "\t" << b[i].finrow << "\t" << b[i].fincol << "\t" << b[i].frow << "\t" << \
        b[i].fcol << "\t" << b[i].Ksize << "\t" << b[i].Kstride << "\t" << b[i].pad << "\t" << b[i].mask << "\t" << b[i].addr_in << "\t" << \
        b[i].addr_wght << "\t" << b[i].addr_out << "\t" << b[i].pool << "\t" << b[i].relu << "\t" << 0 << "\t" << std::endl; 
    } 
}

vector< layer > CNN4FPGA::EstabNet4FPGA(vector< lycfg> layer_cfg, int* wght_length, int* layerdef_length, int* fm_length, int* infm_length, int* lastfm_length) {
    //padding
    lycfg intmLayer;
    vector< lycfg > intmRes;
    int orig = layer_cfg.size();

    for(int i=0; i<orig; i++){
       intmLayer.type         = layer_cfg[i].type;
       intmLayer.num_input    = ((layer_cfg[i].num_input+DUM-1)/DUM)*DUM;
       intmLayer.num_output   = ((layer_cfg[i].num_output+DUM-1)/DUM)*DUM; 
       intmLayer.input_height = layer_cfg[i].input_height; 
       intmLayer.input_width  = layer_cfg[i].input_width;
       intmLayer.kernel_size  = layer_cfg[i].kernel_size;
       intmLayer.kernel_stride= layer_cfg[i].kernel_stride;
       intmLayer.kernel_pad   = layer_cfg[i].kernel_pad;
       intmLayer.output_height= layer_cfg[i].output_height;
       intmLayer.output_width = layer_cfg[i].output_width;
       intmRes.push_back(intmLayer);
    }
    int rs = intmRes.size()-1;
    *lastfm_length = intmRes[rs].num_output*intmRes[rs].output_height*intmRes[rs].output_width;
    *infm_length = intmRes[0].num_input*\
                   (intmRes[0].input_height-2*intmRes[0].kernel_pad)*\
                   (intmRes[0].input_width-2*intmRes[0].kernel_pad);

    //transform to fpga model layer
    vector< layer > result;
    int fm_addr = 0;
    for(int i=0; i<orig; i++){
        layer r = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        if(intmRes[i].type.compare("conv")==0){
            r.fin = intmRes[i].num_input; 
            r.fout = intmRes[i].num_output; 
            r.finrow = intmRes[i].input_height; 
            r.fincol = intmRes[i].input_width; 
            r.frow = intmRes[i].output_height; 
            r.fcol = intmRes[i].output_width; 
            r.Ksize = intmRes[i].kernel_size; 
            r.Kstride = intmRes[i].kernel_stride; 
            r.pad = intmRes[i].kernel_pad; 
            r.mask = DUM; 
            if(i<orig-1){
                if((intmRes[i+1].type.compare("conv")!=0)) {
                    if(intmRes[i+1].type.compare("relu")==0) r.relu = 1; 
                    else if(intmRes[i+1].type.compare("pool")==0) r.pool = 1; 
                    if(i<orig-2){
                       if((intmRes[i+2].type.compare("conv")!=0)) {
                           if(intmRes[i+2].type.compare("relu")==0) r.relu = 1; 
                           else if(intmRes[i+2].type.compare("pool")==0) r.pool = 1; 
                       }
                    }
                }
            }
            if(i==0){
                fm_addr = r.fin*(r.finrow-2*r.pad)*(r.fincol-2*r.pad);
            }
            fm_addr += r.fout* r.frow* r.fcol;
            result.push_back(r);
        }
    }
    *fm_length = fm_addr;
    int wght_addr = 0;
    for(int i=0; i<result.size(); i++) {
        result[i].addr_wght = wght_addr;
        wght_addr += (result[i].fin* result[i].fout* result[i].Ksize* result[i].Ksize + result[i].fout);
    } 
    *wght_length = wght_addr;
    *layerdef_length = (result.size()+1)*16;

    fm_addr = 0;
    result[0].addr_in = fm_addr + wght_addr;
    fm_addr += result[0].fin* result[0].frow* result[0].fcol;
    result[0].addr_out = fm_addr + wght_addr;
    for(int i=1; i<result.size(); i++) {
        result[i].addr_in = result[i-1].addr_out;
        fm_addr += result[i-1].fout* result[i-1].frow* result[i-1].fcol;
        result[i].addr_out = fm_addr + wght_addr;
    } 
    return result;
}

vector< lycfg > CNN4FPGA::ParseCNN_Arch(shared_ptr<Net<float> > net_) {
  //--- left for future use: input data shape and batch info from caffe ---//
  //Blob< float >* in_blob = net_->input_blobs()[0];
  //vector< int > shape = in_blob->shape(); // int shape[0] batch, shape[1] channel, shape[2] height

  //Please be noticed that 'layers for weights' in 'net->layers'
  //and 'layers for feature maps' in 'net->blobs' are different.
  //ReLu layer does not use 'net->blobs'
  vector< lycfg > result; //vector of layers parsed
  vector< shared_ptr< Layer< float > > > layers = net_->layers();      //net layers for weight kernel info and values (blob)
  vector< shared_ptr< Blob< float > > > vfmlayer = net_->blobs();      //net blobs for feature map kernel info and values
  vector< string > layer_name = net_->layer_names();
  vector< int > num;
  int kk=0;
  printf("total layers %d:\n", layer_name.size());
  for(int k=0; k<layer_name.size(); k++){
    num = vfmlayer[kk]->shape();
    LayerParameter lparam = layers[k]->layer_param();
    if( layer_name[k].compare(0, 2, "fc")==0 ) {
        break;
    }
    else if( layer_name[k].compare(0, 4, "conv")==0 ) {
        ConvolutionParameter cparam = lparam.convolution_param(); 
        lycfg t;
        t.type = "conv";
        t.num_input = (int)num[1]; 
        t.num_output = (int)(cparam.num_output());
        t.input_height =  num[2] + 2*(int)(cparam.pad(0));
        t.input_width =  num[3] + 2*(int)(cparam.pad(0));
        t.kernel_size =  (int)cparam.kernel_size(0);
        //t.kernel_stride =  (int)(cparam.stride(0));
        t.kernel_stride =  1;
        t.kernel_pad =  (int)(cparam.pad(0));
        t.output_height =  (t.input_height-t.kernel_size+t.kernel_stride)/t.kernel_stride;
        t.output_width =  (t.input_width-t.kernel_size+t.kernel_stride)/t.kernel_stride;

        result.push_back(t);
        kk=kk+1;
    }
    else if( layer_name[k].compare(0, 4, "pool")==0 ) {
        PoolingParameter pparam = lparam.pooling_param(); 
        lycfg t;
        t.type = "pool";
        t.num_input = (int)num[1]; 
        t.num_output = (int)num[1]; //(int)(pparam.num_output()); temporary workaround;
        t.input_height =  num[2] + 2*(int)(pparam.pad());
        t.input_width =  num[3] + 2*(int)(pparam.pad());
        t.kernel_size =  (int)pparam.kernel_size();
        t.kernel_stride =  (int)(pparam.stride());
        t.kernel_pad =  (int)(pparam.pad());
        t.output_height =  (t.input_height-t.kernel_size+t.kernel_stride)/t.kernel_stride;
        t.output_width =  (t.input_width-t.kernel_size+t.kernel_stride)/t.kernel_stride;
        result.push_back(t);
        kk=kk+1;
    }
    else if( layer_name[k].compare(0, 4, "relu")==0 ) {
        lycfg t;
        t.type = "relu";
        t.num_input = result[k-1].num_output; 
        t.num_output = result[k-1].num_output;
        t.input_height = result[k-1].output_height;
        t.input_width = result[k-1].output_width;
        t.kernel_size = 1;
        t.kernel_stride = 1;
        t.kernel_pad = 0;
        t.output_height = result[k-1].output_height;
        t.output_width = result[k-1].output_width;
        result.push_back(t);
    }
  }
  printf("finish cnn parsing\n");
  return result;
}

void CNN4FPGA::print_layer(vector< layer > net4fpga) {
//add mask    std::cout << "fin\tfout\tfinrow\tfincol\tfrow\tfcol\tKsize\tKstride\tpad\tmask\taddr_in\taddr_wght\taddr_out\tpool\trelu\t0" << std::endl;
    std::cout << "fin\tfout\tfinrow\tfincol\tfrow\tfcol\tKsize\tKstride\tpad\taddr_wght\taddr_in\taddr_out\tpool\trelu\t0" << std::endl;
    for(int i=0; i<net4fpga.size(); i++){
        std::cout \
        << " type: " << "conv" \
        << " in: " << net4fpga[i].fin \
        << " out: " << net4fpga[i].fout \
        << " in_hei: " << net4fpga[i].finrow \ 
        << " in_wid: " << net4fpga[i].fincol \
        << " out_hei: " << net4fpga[i].frow \
        << " out_wid: " << net4fpga[i].fcol \
        << " kernel: " << net4fpga[i].Ksize \
        << " stride: " << net4fpga[i].Kstride \
        << " pad: " << net4fpga[i].pad \
        << " addr_wght: " << net4fpga[i].addr_wght \
        << " addr_in: " << net4fpga[i].addr_in \
        << " addr_out: " << net4fpga[i].addr_out \
        << " pool: " << net4fpga[i].pool \
        << " relu: " << net4fpga[i].relu \
        << std::endl;
    } 
}

void CNN4FPGA::print_lycfg(vector< lycfg > origin){
    for(int i=0; i<origin.size(); i++){
       std::cout \
       << " type: " << origin[i].type \
       << " in: " << origin[i].num_input \
       << " out: " << origin[i].num_output \
       << " in_hei: " << origin[i].input_height \
       << " in_wid: " << origin[i].input_width \
       << " out_hei: " << origin[i].output_height \
       << " out_wid: " << origin[i].output_width \
       << " kernel: " << origin[i].kernel_size \
       << " stride: " << origin[i].kernel_stride \
       << " pad: " << origin[i].kernel_pad \
       << std::endl;
    }       
}

void CNN4FPGA::verify_reorder(float* wght_dram){

    ofstream curr;
    curr.open("curr.txt");
    for(int c=0; c<wght_length; c++){
        curr << wght_dram[c] << endl;
    }
    curr.close();
   int i=0;
   ifstream gd, ths;
   gd.open("goldedn_wght.txt");
   string gdline;
   //ths.open("curr_before.txt");
   //string thsline;
   if (gd.is_open())
   {
     while ( getline(gd, gdline) )
     {
        //getline(ths, thsline);
        //float tmpt = std::strtof(thsline.c_str(), 0); 
         float tmpd = std::strtof(gdline.c_str(), 0); 
         if((rcmp(tmpd, wght_dram[i])>1e-5)&&(i>=0)){
           std::cout << "wrong numbers in weight?" << std::endl; 
           std::cout << "positions: " << i << " orig: " << tmpd << " new: " << wght_dram[i] << std::endl; 
           break;
         }
         i++;
     }
     gd.close();
     //ths.close();
   }
   else cout << "Unable to open file";
}


class OpenCLFPGAModel {

public:
  OpenCLFPGAModel(){
     initialized = false;
  } 
  
  void setFPGAModel(const char* bin_path, CNN4FPGA cnnModel);
  void FPGAinit();    
  vector< float > FPGAexec(float* image);    

  ~OpenCLFPGAModel() {
    if (initialized) {
      clReleaseProgram(program);
      clReleaseKernel(kernel);
      clReleaseCommandQueue(commands);
      clReleaseContext(context);
      clReleaseMemObject(DRAM);
      clReleaseMemObject(DRAM_LY);
      free(wght_dram);
      free(ly_host);
#if SIM
      free(DRAM_sim);
      free(DRAM_LY_sim);
#endif;
    }
  }

  cl_context& getContext() {
    if (initialized) {
      return context;
    }
    else {
      throw std::runtime_error("environment not setup");
    }
  }

  cl_command_queue& getCmdQueue() {
    if (initialized) {
      return commands;
    }
    else {
      throw std::runtime_error("environment not setup");
    }
  }

  cl_kernel& getKernel() {
    if (initialized) {
      return kernel;
    }
    else {
      throw std::runtime_error("environment not setup");
    }
  }
  cl_mem& getArg1() {
    if (initialized) {
      return DRAM_LY;
    }
    else {
      throw std::runtime_error("environment not setup");
    }
  }
  cl_mem& getArg0() {
    if (initialized) {
      return DRAM;
    }
    else {
      throw std::runtime_error("environment not setup");
    }
  }

private:

  int load_file(
      const char *filename, 
      char **result)
  { 
    int size = 0;
    FILE *f = fopen(filename, "rb");
    if (f == NULL) 
    { 
      *result = NULL;
      return -1; // -1 means file opening fail 
    } 
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *result = (char *)malloc(size+1);
    if (size != fread(*result, sizeof(char), size, f)) 
    { 
      free(*result);
      return -2; // -2 means file reading fail 
    } 
    fclose(f);
    (*result)[size] = 0;
    return size;
  }
  template<typename dd_type, typename ds_type> 
    void mycpy(dd_type* dstdram, int dst, ds_type* srcdram, int src, int length);
  template<typename d_type> 
    void reorder_output(d_type *m, int num, int row, int col);
  template<typename d_type>
    void prepare_image(d_type *m, int addr_m); 
  void reorder_image_verify(float* ddram);

private:
  bool initialized;
  int ly_vol;
  int wt_vol;
  int fm_vol;
  int infm_vol;
  int lastfm_vol;
  int dram_vol;
  float* wght_dram;
  int* ly_host;
  vector< lycfg > orig;
  vector< layer > cnn4fpga;

  cl_context context;                 // compute context
  cl_command_queue commands;          // compute command queue
  cl_program program;                 // compute program
  cl_kernel kernel;                   // compute kernel
  cl_mem DRAM;//[DATA_VOL];              // device memory used for the input/output data
  cl_mem DRAM_LY;//[LayerDef];              // device memory used for layer definition

  bw_t* DRAM_sim;
  ly_t* DRAM_LY_sim;

};

void OpenCLFPGAModel::setFPGAModel( const char* bin_path, CNN4FPGA cnnModel)
{
      // start platform setting up
      int err;

      ly_vol = cnnModel.layerdef_len();
      wt_vol = cnnModel.weight_len();
      fm_vol = cnnModel.fm_len();
      infm_vol = cnnModel.infm_len();
      lastfm_vol = cnnModel.lastfm_len();
      dram_vol = wt_vol + fm_vol;
      wght_dram = (float*)malloc(sizeof(float)*wt_vol);
      float* wght_tmp = cnnModel.weight_ptr();
      memcpy((void*)wght_dram, (const void*)wght_tmp, sizeof(float)*wt_vol);
      ly_host = (int*)malloc(sizeof(int)*ly_vol);
      int* ly_tmp = cnnModel.layerdef_ptr();
      memcpy((void*)ly_host, (const void*)ly_tmp, sizeof(float)*ly_vol);

      orig = cnnModel.origin;
      cnn4fpga = cnnModel.net4fpga;
      std::cout <<"layer size: "<< ly_vol << std::endl;
      for(int i=0; i<14*16; i++){
        std::cout << ly_host[i] << "\t"; 
        if((i+1)%16==0) 
            std::cout << std::endl;
      }

      const char* kernel_name = "vgg16" ;
      cl_platform_id platform_id;
      cl_device_id device_id;

      char cl_platform_vendor[1001];
      char cl_platform_name[1001];

      // Connect to first platform
      err = clGetPlatformIDs(1, &platform_id, NULL);

      if (err != CL_SUCCESS) {
          throw std::runtime_error(
              "Failed to find an OpenCL platform!");
      }

      err = clGetPlatformInfo(
          platform_id, 
          CL_PLATFORM_VENDOR, 
          1000, 
          (void *)cl_platform_vendor,NULL);

      if (err != CL_SUCCESS) {
          throw std::runtime_error(
              "clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!");
      }

      err = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME,1000,(void *)cl_platform_name,NULL);
      if (err != CL_SUCCESS) {
          throw std::runtime_error("clGetPlatformInfo(CL_PLATFORM_NAME) failed!");
      }

      // Connect to a compute device
      err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id, NULL);
      if (err != CL_SUCCESS) {
          throw std::runtime_error("Failed to create a device group!");
      }

      // Create a compute context 
      context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
      if (!context) {
          throw std::runtime_error("Failed to create a compute context!");
      }

      // Create a command commands
      commands = clCreateCommandQueue(context, device_id, 0, &err);
      if (!commands) {
          throw std::runtime_error("Failed to create a command queue context!");
      }

      // Create Program Objects
      // TODO: this part should not be static, the program
      // should be configurable at runtime

      // Load binary from disk
      unsigned char *kernelbinary;
      int n_i = load_file(bin_path, (char **) &kernelbinary);

      if (n_i < 0) {
          throw std::runtime_error(
              "failed to load kernel from xclbin");
      }
      size_t n_t = n_i;

      int status = 0;

      // Create the compute program from offline
      program = clCreateProgramWithBinary(context, 1, &device_id, &n_t,
              (const unsigned char **) &kernelbinary, &status, &err);
      if ((!program) || (err!=CL_SUCCESS)) {
          throw std::runtime_error(
              "Failed to create compute program from binary");
      }

      // Build the program executable
      err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
      if (err != CL_SUCCESS) {
          throw std::runtime_error("Failed to build program executable!");
      }

      // Create the compute kernel in the program we wish to run
      kernel = clCreateKernel(program, kernel_name, &err);
      if (!kernel || err != CL_SUCCESS) {
          throw std::runtime_error("Failed to create compute kernel!");
      }

      //-------init DRAM-------------//
      DRAM = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(data_t) * dram_vol, NULL, NULL);
      DRAM_LY = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(int) * ly_vol, NULL, NULL);
      if ((!DRAM) || (!DRAM_LY))
      {
        printf("Error: Failed to allocate device memory!\n");
        printf("Test failed\n");
      }

      err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &DRAM);
      err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &DRAM_LY);
      if (err != CL_SUCCESS)
      {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        printf("Test failed\n");
      }

      initialized = true;
      printf("successfully initialize FPGA bitstream!\n");
}

void OpenCLFPGAModel::FPGAinit(){
    //int wt_vol = weight_len();
    //int fm_vol = fm_len();
    //int ly_vol = layerdef_len(); 
    printf("start FPGA init\n");
    if_wght_t *fdram = (if_wght_t*)malloc(sizeof(if_wght_t)* wt_vol);
/*
    ofstream curr;
    curr.open("weight.txt");
    for(int i=0; i<wt_vol; i++){
        curr << wght_dram[i] << std::endl; 
    }
    curr.close();
*/
    mycpy<if_wght_t, float>(fdram, 0, wght_dram, 0, wt_vol);
    printf("finish mycpy FPGA init\n");

#if SIM
  DRAM_sim = (bw_t*)malloc(sizeof(if_data_t)*(wt_vol + fm_vol));
  DRAM_LY_sim = (ly_t*)malloc(sizeof(int)*ly_vol);
  memcpy((void*) DRAM_sim, (const void*)fdram, sizeof(if_wght_t)* wt_vol);
  memcpy((void*) DRAM_LY_sim, (const void*)ly_host, sizeof(int)*ly_vol);

#else

    cl_event event;
    struct timeval t0, t2;
    int err = 0;
    err = clEnqueueWriteBuffer(commands, DRAM, CL_TRUE, 0, sizeof(if_data_t) * wt_vol, fdram, 0, NULL, &event);
    err = clEnqueueWriteBuffer(commands, DRAM_LY, CL_TRUE, 0, sizeof(int) * ly_vol, ly_host, 0, NULL, &event);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array a!\n");
        printf("Test failed\n");
    }

    clWaitForEvents(1, &event);
    gettimeofday(&t0, NULL);
    //executing vgg16 kernels
    printf("write weight& configuration to Device!\n");
    err = clEnqueueTask(commands, kernel, 0, NULL, &event);
      if (err != CL_SUCCESS)
    {
        printf("Error: Failed run kernel!\n");
        printf("Test failed\n");
    }
    clWaitForEvents(1, &event);
    gettimeofday(&t2, NULL);
    float time_kernel = (t2.tv_sec-t0.tv_sec)*1e+3 + (t2.tv_usec-t0.tv_usec)*1e-03 ;
    printf("Weight Transfer time :%8.6f ms\n ", time_kernel);

#endif
  free(fdram);
}

vector< float > OpenCLFPGAModel::FPGAexec(float* image) {

  float *ddram = (float*)malloc(sizeof(float)*infm_vol);
  memcpy((void*)ddram, (const void*)image, sizeof(float)*infm_vol);
  prepare_image<float>(ddram, 0);
  //reorder_image_verify(ddram);
  printf("finish prepare image\n");
  if_data_t *rdram = (if_data_t*)malloc(sizeof(if_data_t)*infm_vol);
  mycpy<if_data_t, float>(rdram, 0, ddram, 0, infm_vol);

  struct timeval t0, t1, t2, t3, t4, t5;

#if SIM

  int DATA_OFFSET = (cnn4fpga[0].addr_in)*sizeof(if_wght_t)/sizeof(bw_t);
  memcpy((void*)(DRAM_sim+DATA_OFFSET), (const void*)rdram, sizeof(if_data_t)*infm_vol);

  printf("start CNN computation\n");
  //TODO: CNN simulation kernel here
  vgg16(DRAM_sim, DRAM_LY_sim);

  printf("finish CNN computation\n");

  if_data_t* feat_dev = (if_data_t*) malloc(sizeof(if_data_t)*(lastfm_vol)); 
  float* fpga_feat = (float*) malloc(sizeof(float)*(lastfm_vol)); 
  std::cout<< "last feature map vollum: "<< lastfm_vol <<std::endl;

  DATA_OFFSET = sizeof(if_data_t)*(cnn4fpga[cnn4fpga.size()-1].addr_out)/sizeof(bw_t);
  memcpy((void*)feat_dev, (const void*)(DRAM_sim+DATA_OFFSET), sizeof(if_data_t)*lastfm_vol);
  mycpy<float, if_data_t>(fpga_feat, 0, feat_dev, 0, lastfm_vol);
  const int num = orig[orig.size()-1].num_output;
  const int row = orig[orig.size()-1].output_height;
  const int col = orig[orig.size()-1].output_width;
  reorder_output<float>(fpga_feat, num, row, col);

  vector< float > result;

  for(int i=0; i<lastfm_vol; i++){
    result.push_back(fpga_feat[i]); 
  }
  return result;

  free(feat_dev);
  free(fpga_feat);
  free(ddram);
  free(rdram);

#else

  int err=0;

  cl_event event;
  int DATA_OFFSET = cnn4fpga[0].addr_in*sizeof(if_wght_t);
  err = clEnqueueWriteBuffer(commands, DRAM, CL_TRUE, DATA_OFFSET, sizeof(data_t)*infm_vol, rdram, 0, NULL, &event);
  if (err != CL_SUCCESS)
  {
      printf("Error: Failed to write to source array a!\n");
      printf("Test failed\n");
  }

  clWaitForEvents(1, &event);
  gettimeofday(&t0, NULL);
  //executing vgg16 kernels
  printf("start kernel!\n");
  err = clEnqueueTask(commands, kernel, 0, NULL, &event);
  if (err != CL_SUCCESS)
  {
      printf("Error: Failed run kernel!\n");
      printf("Test failed\n");
  }
  clWaitForEvents(1, &event);
  gettimeofday(&t2, NULL);
  float time_kernel = (t2.tv_sec-t0.tv_sec)*1e+3 + (t2.tv_usec-t0.tv_usec)*1e-03 ;
  printf("FPGA time :%8.6f ms\n ", time_kernel);

  float* fpga_feat = (float*) malloc(sizeof(float)*(lastfm_vol)); 
  if_data_t* feat_dev = (if_data_t*) malloc(sizeof(if_data_t)*(lastfm_vol)); 
  DATA_OFFSET = sizeof(if_data_t)*(cnn4fpga[cnn4fpga.size()-1].addr_out);
  err = clEnqueueReadBuffer( commands, DRAM, CL_TRUE, DATA_OFFSET, sizeof(if_data_t)*lastfm_vol, feat_dev, 0, NULL, &event );
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
  }
  clWaitForEvents(1, &event);
  mycpy<float, if_data_t>(fpga_feat, 0, feat_dev, 0, lastfm_vol);
  const int num = orig[orig.size()-1].num_output;
  const int row = orig[orig.size()-1].output_height;
  const int col = orig[orig.size()-1].output_width;
  reorder_output<float>(fpga_feat, num, row, col);

  vector< float > result;

  for(int i=0; i<lastfm_vol; i++){
    result.push_back(fpga_feat[i]); 
  }
  return result;

  free(feat_dev);
  free(fpga_feat);

#endif
// if_data_t* layer1_fm = (if_data_t*)malloc(sizeof(float)*FM11);

}

template<typename d_type>
void OpenCLFPGAModel::prepare_image(d_type *m, int addr_m) {

	d_type adata[UNROLL][224][224];
	for(int i=0; i<3; i++) {
	    for(int j=0; j<224; j++) {
	        for(int k=0; k<224; k++){
	                adata[i][j][k] = m[i*224*224+j*224+k + addr_m];
	        }
	    }
	}
	for(int i=3; i<UNROLL; i++) {
	    for(int j=0; j<224; j++) {
	        for(int k=0; k<224; k++){
	            adata[i][j][k] = (d_type)0;
	        }
	    }
	}
	for(int j=0; j<224; j++) {
	    for(int k=0; k<224; k++){
	        for(int i=0; i<UNROLL; i++) {
	            m[(j*224+k)*UNROLL + i + addr_m] = adata[i][j][k];
	        }
	    }
	}
}


template<typename dd_type, typename ds_type>
void OpenCLFPGAModel::mycpy(dd_type* dstdram, int dst, ds_type* srcdram, int src, int length){
    int nancnt = 0;
    for(int i=0; i<length; i++) {
        if(isnan(srcdram[i+ src])) {
            dstdram[i+ dst] = (dd_type)4096;
            nancnt++;
        }
        else
            dstdram[i+ dst] = (dd_type)(srcdram[i+ src]);
    }
    printf("nancnt: %d, \n", nancnt);
}

template<typename d_type>
void OpenCLFPGAModel::reorder_output(d_type *m, int num, int row, int col) {
   //static d_type data[num][row][col];
   d_type* data = (d_type*)malloc(sizeof(d_type)*num*row*col);
   for(int ii=0; ii<num; ii+=UNROLL) {
       for(int r=0; r<row; r++) {
           for(int c=0; c<col; c++) {
               for(int i=0; i<UNROLL; i++) {
                   data[(ii+i)*row*col + r*col + c] = m[(r*col+c)*UNROLL + (ii)*row*col + i];
               }
           }
       }
   }
   for(int i=0; i<num; i++) {
       for(int j=0; j<row; j++) {
           for(int k=0; k<col; k++) {
               m[i*row*col + j*col + k] = data[i*col*row + j*col + k];
           }
       }
   }
   free(data);
}

void OpenCLFPGAModel::reorder_image_verify(float* ddram){

  ofstream curr;
  curr.open("input_image.txt", ios::app);
  for(int i=0; i<infm_vol; i++){
    curr << ddram[i] << std::endl; 
  } 
  curr.close();

  ifstream icur;
  icur.open("input_image_gold.txt");
  string line;
  int j=0;
   if (icur.is_open())
   {
     while ( getline(icur, line) )
     {
         float tmpd = std::strtof(line.c_str(), 0); 
         if((rcmp(tmpd, ddram[j])>1e-5)){
           std::cout << "wrong numbers in input images?" << std::endl; 
           std::cout << "positions: " << j << " orig: " << tmpd << " new: " << ddram[j] << std::endl; 
           break;
         }
         j++;
     }
   }
   else cout << "Unable to open file";
}
