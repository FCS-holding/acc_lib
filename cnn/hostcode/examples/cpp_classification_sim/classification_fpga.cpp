#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <CL/opencl.h>
#include <sys/time.h>
//#include "/curr/chenzhang/tool/caffe/FPGA/include/kernel-cl-batch.hpp"
//#include "/curr/chenzhang/tool/caffe_fpga/FPGA/include/OpenCLEnv.h"

#include "vgg16_sw2.hpp"
#include "/curr/chenzhang/tool/caffe_fpga/FPGA/include/falconMLlib.h"


#ifdef USE_OPENCV
// NOLINT(build/namespaces)
using namespace caffe;  
using std::string;

#define USE_FPGA 1
#define FPGA_Verify 0
#define IMSHOW 0

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier{
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);

  //std::vector<Prediction> Classify(const cv::Mat& img, int N = 5, bw_t* DRAM = 0, ly_t* DRAM_LY = 0 );
  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);
 
 public:
  CNN4FPGA cnn_model;
  OpenCLFPGAModel fpga;

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);
  //std::vector<float> FPGA_Predict(const cv::Mat& img, bw_t* DRAM, ly_t* DRAM_LY);
  std::vector<float> FPGA_Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  //CNN4FPGA cnn_model(net_);
  cnn_model.setCNNModel(net_);
  fpga.setFPGAModel("/curr/chenzhang/tool/caffe_fpga/FPGA/xclbin_sda2015.3/vgg16_fxdyn.xclbin", cnn_model);
  fpga.FPGAinit();


  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
//std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N , bw_t* DRAM, ly_t* DRAM_LY){
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N){
#if USE_FPGA
  //std::vector<float> output = FPGA_Predict(img, DRAM, DRAM_LY);
  std::vector<float> output = FPGA_Predict(img);
#else
  std::vector<float> output = Predict(img);
#endif

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  timeval startSw, endSw;
  gettimeofday(&startSw, NULL);
  /*library implementation*/
  net_->ForwardPrefilled(); 
  /*library implementation*/
  gettimeofday(&endSw, NULL);
  printf("library time :%8.6f ms\n ", (endSw.tv_sec-startSw.tv_sec)*1e+3 + (endSw.tv_usec-startSw.tv_usec)*1e-03 );

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

std::vector<float> Classifier::FPGA_Predict(const cv::Mat& img){
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);

  // Forward dimension change to all layers.
  net_->Reshape();
  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);
  Preprocess(img, &input_channels);

  //FPGA Acceleration Test
  printf("FPGA Prediction Start\n");

  float fpga_result[1000];

  //int DATA_VOL = (LayerDef+ WEIGHTSIZE+ INFM + FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43+ FM51+FM52+FM53+PL53);
  //int WT_VOL = (LayerDef+ WEIGHTSIZE+ INFM);
  float *ddram = (float*)malloc(sizeof(float)*INFM);
  const float* fdata = net_->input_blobs()[0]->cpu_data();
  uchar* fatdata = input_channels.at(0).data;
  for(int i=0; i<INFM; i++) {
    ddram[i] = (float)fdata[i];
  }

/*
  printf("write dram with input feature map\n");
  prepare_image<float>(ddram, 0);

  if_data_t *dddram = (if_data_t*)malloc(sizeof(if_data_t)*INFM);
  mycpy<if_data_t, float>(dddram, 0, ddram, 0, INFM);

  struct timeval t0, t1, t2, t3, t4, t5;

  int DATA_OFFSET = (WEIGHTSIZE)*sizeof(if_data_t)/sizeof(bw_t);
  memcpy((void*)(DRAM+DATA_OFFSET), (const void*)dddram, sizeof(if_data_t) * INFM);

  //TODO: CNN simulation kernel here
  vgg16(DRAM, DRAM_LY);

  printf("finish computation\n");

  if_data_t* feat_dev = (if_data_t*) malloc(sizeof(if_data_t)*(PL53)); 
  float* fpga_feat = (float*) malloc(sizeof(float)*(PL53)); 

  DATA_OFFSET = sizeof(if_data_t)*(WEIGHTSIZE+ INFM + FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43+ FM51+FM52+FM53)/sizeof(bw_t);
  memcpy((void*)feat_dev, (const void*)(DRAM+DATA_OFFSET), sizeof(if_data_t)*PL53);
  mycpy<float, if_data_t>(fpga_feat, 0, feat_dev, 0, PL53);

// if_data_t* layer1_fm = (if_data_t*)malloc(sizeof(float)*FM11);
// DATA_OFFSET = sizeof(if_data_t)*(WEIGHTSIZE+ INFM)/sizeof(bw_t);
// memcpy((void*)layer1_fm, (const void*)(DRAM+DATA_OFFSET), sizeof(if_data_t)*FM11);
// FILE* ly1_file = fopen("layer1_fm.txt" ,"w");
// int fcnt = 0;
// for(int t=0; t<FM11; t++){
// float l1data = (float)layer1_fm[t];
//   if( l1data != 0.0) 
//       fcnt++;
//   fprintf(ly1_file, "%f\n", l1data);
// }
// printf("non zero counts: %d\n", fcnt);
// fclose(ly1_file);
// free(layer1_fm);

  //gettimeofday(&t2, NULL);
  //memcpy(fpga_feat, (const void*)(fdram+DATA_VOL-PL53), sizeof(float)*PL53);
  reorder_output<float, POOL53, POOL53_R, POOL53_R, HWFR, HWFC>(fpga_feat);
  //gettimeofday(&t3, NULL);
  //printf("caffe FPGA time :%8.6f ms\n ", (t2.tv_sec-t1.tv_sec)*1e+3 + (t2.tv_usec-t1.tv_usec)*1e-03 );
  //printf("caffe FPGA+reorder time :%8.6f ms\n ", (t3.tv_sec-t1.tv_sec)*1e+3 + (t3.tv_usec-t1.tv_usec)*1e-03 );
*/

  vector< float > result = fpga.FPGAexec(ddram);
  float* fpga_feat = (float*) malloc(sizeof(float)*result.size()); 
  for(int i=0; i<result.size(); i++){
    fpga_feat[i] = result[i]; 
  }
  ann(fpga_feat, fpga_result);
  //printf("FPGA exec finish\n");

#if FPGA_Verify
  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  printf("total %d of output_layer channels\n", output_layer->channels());

  FILE *rp = fopen("/curr/chenzhang/tool/caffe/result.txt", "w");
  for(int k =0; k<1000; k++) {
    fprintf(rp, "caffe: %f,\t software:%f\t FPGA:%f\n", begin[k], result[k], fpga_result[k]);
    //fprintf(rp, "caffe: %f,\t software:%f\t, diff: %f\n", begin[k], result[k], begin[k]-result[k]);
  } 
  fclose(rp); 
  printf("output_layer->channels() = %d\n", output_layer->channels());

  return std::vector<float>(begin, end);
#else
  Blob<float>* output_layer = net_->output_blobs()[0];
  //const float* begin = output_layer->cpu_data();
  //const float* end = begin + output_layer->channels();

  std::vector<float> output;
  for(int k =0; k<output_layer->channels(); k++) {
    output.push_back(fpga_result[k]); 
  }

  free(ddram);
  //free(dddram);
  //free(feat_dev);
  free(fpga_feat);
  printf("FPGA Prediction finish\n");
  return output;
#endif

}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

//OpenCLEnv3 env("/curr/chenzhang/tool/caffe_fpga/FPGA/xclbin_sda2015.3/vgg16_fxdyn.xclbin", "vgg16", LayerDef, (WEIGHTSIZE+ INFM + FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43+ FM51+FM52+FM53+PL53) );
//
int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file   = argv[4];
  Classifier classifier(model_file, trained_file, mean_file, label_file);

/*
  int DATA_VOL = (WEIGHTSIZE);
  float *wght_dram = (float*)malloc(sizeof(float)*DATA_VOL);
  memset(wght_dram, 0, sizeof(float)*DATA_VOL); 
  init_dram<float>(wght_dram);
  //for(int k=0; k<CONV11; k++){
  //  wght_dram[k+ Layer11-CONV11] = wght_dram[k+ Layer11-CONV11]/256.0;
  //}
  prepare_weight<float>(wght_dram);
  system("CLS");
  system("clear");


  int ly_host[14*16] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 13,
  conv1_1.fin, conv1_1.fout, conv1_1.finrow, conv1_1.fincol, conv1_1.frow, conv1_1.fcol, conv1_1.Ksize, conv1_1.Kstride,  
    conv1_1.pad, conv1_1.mask, conv1_1.addr_in, conv1_1.addr_wght, conv1_1.addr_out, conv1_1.pool, conv1_1.relu, 0x00000000, //0x00000000, 
  conv1_2.fin, conv1_2.fout, conv1_2.finrow, conv1_2.fincol, conv1_2.frow, conv1_2.fcol, conv1_2.Ksize, conv1_2.Kstride,  
    conv1_2.pad, conv1_2.mask, conv1_2.addr_in, conv1_2.addr_wght, conv1_2.addr_out, conv1_2.pool, conv1_2.relu, 0x00000000, //0x10041004, 
  conv2_1.fin, conv2_1.fout, conv2_1.finrow, conv2_1.fincol, conv2_1.frow, conv2_1.fcol, conv2_1.Ksize, conv2_1.Kstride,  
    conv2_1.pad, conv2_1.mask, conv2_1.addr_in, conv2_1.addr_wght, conv2_1.addr_out, conv2_1.pool, conv2_1.relu, 0x00000000, //0x00001004, 
  conv2_2.fin, conv2_2.fout, conv2_2.finrow, conv2_2.fincol, conv2_2.frow, conv2_2.fcol, conv2_2.Ksize, conv2_2.Kstride,  
    conv2_2.pad, conv2_2.mask, conv2_2.addr_in, conv2_2.addr_wght, conv2_2.addr_out, conv2_2.pool, conv2_2.relu, 0x00000000, // 0x10021008, 
  conv3_1.fin, conv3_1.fout, conv3_1.finrow, conv3_1.fincol, conv3_1.frow, conv3_1.fcol, conv3_1.Ksize, conv3_1.Kstride,  
    conv3_1.pad, conv3_1.mask, conv3_1.addr_in, conv3_1.addr_wght, conv3_1.addr_out, conv3_1.pool, conv3_1.relu, 0x00000000, // 0x10021010,
  conv3_2.fin, conv3_2.fout, conv3_2.finrow, conv3_2.fincol, conv3_2.frow, conv3_2.fcol, conv3_2.Ksize, conv3_2.Kstride,  
    conv3_2.pad, conv3_2.mask, conv3_2.addr_in, conv3_2.addr_wght, conv3_2.addr_out, conv3_2.pool, conv3_2.relu, 0x00000000, // 0x00001010,
  conv3_3.fin, conv3_3.fout, conv3_3.finrow, conv3_3.fincol, conv3_3.frow, conv3_3.fcol, conv3_3.Ksize, conv3_3.Kstride,  
    conv3_3.pad, conv3_3.mask, conv3_3.addr_in, conv3_3.addr_wght, conv3_3.addr_out, conv3_3.pool, conv3_3.relu, 0x00000000, // 0x00001010,
  conv4_1.fin, conv4_1.fout, conv4_1.finrow, conv4_1.fincol, conv4_1.frow, conv4_1.fcol, conv4_1.Ksize, conv4_1.Kstride,  
    conv4_1.pad, conv4_1.mask, conv4_1.addr_in, conv4_1.addr_wght, conv4_1.addr_out, conv4_1.pool, conv4_1.relu, 0x00000000, // 0x00001010,
  conv4_2.fin, conv4_2.fout, conv4_2.finrow, conv4_2.fincol, conv4_2.frow, conv4_2.fcol, conv4_2.Ksize, conv4_2.Kstride,  
    conv4_2.pad, conv4_2.mask, conv4_2.addr_in, conv4_2.addr_wght, conv4_2.addr_out, conv4_2.pool, conv4_2.relu, 0x00000000, // 0x00001010,
  conv4_3.fin, conv4_3.fout, conv4_3.finrow, conv4_3.fincol, conv4_3.frow, conv4_3.fcol, conv4_3.Ksize, conv4_3.Kstride,  
    conv4_3.pad, conv4_3.mask, conv4_3.addr_in, conv4_3.addr_wght, conv4_3.addr_out, conv4_3.pool, conv4_3.relu, 0x00000000, // 0x00001010,
  conv5_1.fin, conv5_1.fout, conv5_1.finrow, conv5_1.fincol, conv5_1.frow, conv5_1.fcol, conv5_1.Ksize, conv5_1.Kstride,  
    conv5_1.pad, conv5_1.mask, conv5_1.addr_in, conv5_1.addr_wght, conv5_1.addr_out, conv5_1.pool, conv5_1.relu, 0x00000000, // 0x20011008,
  conv5_2.fin, conv5_2.fout, conv5_2.finrow, conv5_2.fincol, conv5_2.frow, conv5_2.fcol, conv5_2.Ksize, conv5_2.Kstride,  
    conv5_2.pad, conv5_2.mask, conv5_2.addr_in, conv5_2.addr_wght, conv5_2.addr_out, conv5_2.pool, conv5_2.relu, 0x00000000, // 0x20041002,
  conv5_3.fin, conv5_3.fout, conv5_3.finrow, conv5_3.fincol, conv5_3.frow, conv5_3.fcol, conv5_3.Ksize, conv5_3.Kstride,  
    conv5_3.pad, conv5_3.mask, conv5_3.addr_in, conv5_3.addr_wght, conv5_3.addr_out, conv5_3.pool, conv5_3.relu, 0x00000000  // 0x00001002
    }; 

  int WT_VOL= ( WEIGHTSIZE );
  //int WT_VOL= ( LayerDef );
  if_wght_t *fdram = (if_wght_t*)malloc(sizeof(if_wght_t)* WT_VOL);
  printf("begin to write weight to if_wght_t\n");
  mycpy<if_wght_t, float>(fdram, 0, wght_dram, 0, WEIGHTSIZE);
  printf("finish to write weight to if_wght_t\n");
  int *ly_dram = (int*)malloc(sizeof(int)*14*16);
  memcpy((void *)(ly_dram), (const void*)ly_host, sizeof(int)*14*16);

  bw_t* DRAM = (bw_t*)malloc(sizeof(if_data_t)*(WEIGHTSIZE+ INFM + FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43+ FM51+FM52+FM53+PL53+PL53));
  ly_t* DRAM_LY = (ly_t*)malloc(sizeof(int)*LayerDef);

  memcpy((void*) DRAM, (const void*)fdram, sizeof(if_wght_t)* WT_VOL);
  memcpy((void*) DRAM_LY, (const void*)ly_dram, sizeof(int)*14*16);

  free(fdram);
  free(wght_dram);
  free(ly_dram);

  std::ifstream out;
  string str = argv[5];
  out.open(str.c_str(), ios::in);
  
  while(1)
  {
    string file ;
    std::getline(out, file);
    if (out.eof()) break;

    cv::Mat img = cv::imread(file, -1);
    CHECK(!img.empty()) << "Unable to decode image " << file;
    std::cout << "---------- Prediction for " << file << " ----------" << std::endl;

    std::vector<Prediction> predictions;
    predictions = classifier.Classify(img, 5, DRAM, DRAM_LY);

    //
    for (size_t i = 0; i < predictions.size(); ++i) {
      Prediction p = predictions[i];
      std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;
    }
  }
  free(DRAM);
  free(DRAM_LY);

*/   

  std::ifstream out;
  string str = argv[5];
  out.open(str.c_str(), ios::in);
  
  while(1)
  {
    string file ;
    std::getline(out, file);
    if (out.eof()) break;

    cv::Mat img = cv::imread(file, -1);
    CHECK(!img.empty()) << "Unable to decode image " << file;
    std::cout << "---------- Prediction for " << file << " ----------" << std::endl;

    std::vector<Prediction> predictions;
    //predictions = classifier.Classify(img, 5, DRAM, DRAM_LY);
    predictions = classifier.Classify(img, 5);

    //
    for (size_t i = 0; i < predictions.size(); ++i) {
      Prediction p = predictions[i];
      std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;
    }
  }

  printf("List Prediction finish\n");
  return 0;

}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV

/*  
  if_data_t* dresult = (if_data_t*)malloc(sizeof(if_data_t)*(FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43+ FM51+FM52+FM53+PL53));
  memcpy((void*)dresult, (const void*)(DRAM+(WEIGHTSIZE+INFM)*sizeof(if_data_t)/sizeof(bw_t)), sizeof(if_data_t)*(FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43+ FM51+FM52+FM53+PL53));
  int addr1  = 0;
  int addr2  = FM11+FM12;
  int addr3  = FM11+FM12+PL12;
  int addr4  = FM11+FM12+PL12 + FM21+FM22;
  int addr5  = FM11+FM12+PL12 + FM21+FM22+PL22;
  int addr6  = FM11+FM12+PL12 + FM21+FM22+PL22+ FM31;
  int addr7  = FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33;
  int addr8  = FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33;
  int addr9  = FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41;
  int addr10 = FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43;
  int addr11 = FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43;
  int addr12 = FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43+ FM51;
  int addr13 = FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43+ FM51+FM52+FM53;
  myprint<if_data_t>(dresult, FM11, addr1 , "FM11.txt");
  myprint<if_data_t>(dresult, PL12, addr2 , "PL12.txt");
  myprint<if_data_t>(dresult, FM21, addr3 , "FM21.txt");
  myprint<if_data_t>(dresult, PL22, addr4 , "PL22.txt");
  myprint<if_data_t>(dresult, FM31, addr5 , "FM31.txt");
  myprint<if_data_t>(dresult, FM32, addr6 , "FM32.txt");
  myprint<if_data_t>(dresult, PL33, addr7 , "PL33.txt");
  myprint<if_data_t>(dresult, FM41, addr8 , "FM41.txt");
  myprint<if_data_t>(dresult, FM42, addr9 , "FM42.txt");
  myprint<if_data_t>(dresult, PL43, addr10, "PL43.txt");
  myprint<if_data_t>(dresult, FM51, addr11, "FM51.txt");
  myprint<if_data_t>(dresult, FM52, addr12, "FM52.txt");
  myprint<if_data_t>(dresult, PL53, addr13, "PL53.txt");
*/



