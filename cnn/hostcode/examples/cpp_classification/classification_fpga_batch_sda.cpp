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
#include "vgg16_sw2.hpp"
//#include "/curr/chenzhang/tool/caffe/FPGA/include/kernel-cl-batch.hpp"
#include "/curr/chenzhang/tool/caffe/FPGA/include/OpenCLEnv.h"

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

#define USE_FPGA 1
#define FPGA_Verify 0
#define IMSHOW 0

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;
OpenCLEnv2 env("/curr/chenzhang/tool/caffe/FPGA/xclbin/vgg16_sda.xclbin", "vgg16", (LayerDef+ WEIGHTSIZE+ INFM + FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43+ FM51+FM52+FM53+PL53) );

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);//, cl_mem DRAM = 0 );

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);
  std::vector<float> FPGA_Predict(const cv::Mat& img);//, cl_mem DRAM);

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
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N ){//, cl_mem DRAM) {
#if USE_FPGA
  std::vector<float> output = FPGA_Predict(img);//, DRAM);
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

std::vector<float> Classifier::FPGA_Predict(const cv::Mat& img){ //, cl_mem DRAM) {
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

  int DATA_VOL = (LayerDef+ WEIGHTSIZE+ INFM + FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43+ FM51+FM52+FM53+PL53);
  int DATA_OFFSET = (LayerDef+ WEIGHTSIZE+ INFM + FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43+ FM51+FM52+FM53);
  int WT_VOL = (LayerDef+ WEIGHTSIZE+ INFM );
  float *ddram = (float*)malloc(sizeof(float)*INFM);
  const float* fdata = net_->input_blobs()[0]->cpu_data();
  uchar* fatdata = input_channels.at(0).data;
  for(int i=0; i<INFM; i++) {
    ddram[i] = (float)fdata[i];
  }
  //printf("write dram with input feature map\n");
  prepare_image(ddram, 0);

  cl_context       context  = env.getContext();
  cl_kernel        kernel   = env.getKernel();
  cl_command_queue commands = env.getCmdQueue();
  cl_mem           DRAM     = env.getArg0();

  /*cl_mem DRAM;//[DATA_VOL];              // device memory used for the input/output data
  DRAM = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(data_t) * DATA_VOL, NULL, NULL);
  if (!DRAM)
  {
    printf("Error: Failed to allocate device memory!\n");
    printf("Test failed\n");
  }
  int err = 0;
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &DRAM);
  err  |= clSetKernelArg(kernel, 1, sizeof(int), &num_layer_host);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to set kernel arguments! %d\n", err);
    printf("Test failed\n");
  }
*/
    int err=0;

  cl_event event;
  struct timeval t0, t1, t2, t3, t4, t5;
  err = clEnqueueWriteBuffer(commands, DRAM, CL_TRUE, (LayerDef)*sizeof(float)+(WEIGHTSIZE)*sizeof(data_t), sizeof(data_t) * INFM, ddram, 0, NULL, &event);

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

  float* fpga_feat = (float*) malloc(sizeof(float)*(PL53)); 
  err = clEnqueueReadBuffer( commands, DRAM, CL_TRUE, sizeof(int) * DATA_OFFSET, sizeof(int) * PL53, fpga_feat, 0, NULL, &event );
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
  }
  clWaitForEvents(1, &event);

  //FPGA Kernel Implementation
  //gettimeofday(&t2, NULL);
  //memcpy(fpga_feat, (const void*)(fdram+DATA_VOL-PL53), sizeof(float)*PL53);
  reorder_output<data_t, POOL53, POOL53_R, POOL53_R, HWFR, HWFC>(fpga_feat);
  //gettimeofday(&t3, NULL);
  //printf("caffe FPGA time :%8.6f ms\n ", (t2.tv_sec-t1.tv_sec)*1e+3 + (t2.tv_usec-t1.tv_usec)*1e-03 );
  //printf("caffe FPGA+reorder time :%8.6f ms\n ", (t3.tv_sec-t1.tv_sec)*1e+3 + (t3.tv_usec-t1.tv_usec)*1e-03 );

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

  int DATA_VOL = (WEIGHTSIZE);
  float *wght_dram = (float*)malloc(sizeof(float)*DATA_VOL);
  memset(wght_dram, 0, sizeof(float)*DATA_VOL); 
  init_dram(wght_dram);
  prepare_weight(wght_dram);
  system("CLS");
  system("clear");

  std::ifstream out;
  string str = argv[5];
  out.open(str.c_str(), ios::in);
  //OpenCLEnv env("/curr/chenzhang/tool/caffe/FPGA/xclbin/vgg16.xclbin", "vgg16");
 
  int ly_host[14*16] ={
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 13,
  conv1_1.fin, conv1_1.fout, conv1_1.finrow, conv1_1.fincol, conv1_1.frow, conv1_1.fcol, conv1_1.Ksize, conv1_1.Kstride,  
    conv1_1.pad, conv1_1.mask, conv1_1.addr_in, conv1_1.addr_wght, conv1_1.addr_out, conv1_1.pool, conv1_1.relu, 0, 
  conv1_2.fin, conv1_2.fout, conv1_2.finrow, conv1_2.fincol, conv1_2.frow, conv1_2.fcol, conv1_2.Ksize, conv1_2.Kstride,  
    conv1_2.pad, conv1_2.mask, conv1_2.addr_in, conv1_2.addr_wght, conv1_2.addr_out, conv1_2.pool, conv1_2.relu, 0, 
  conv2_1.fin, conv2_1.fout, conv2_1.finrow, conv2_1.fincol, conv2_1.frow, conv2_1.fcol, conv2_1.Ksize, conv2_1.Kstride,  
    conv2_1.pad, conv2_1.mask, conv2_1.addr_in, conv2_1.addr_wght, conv2_1.addr_out, conv2_1.pool, conv2_1.relu, 0, 
  conv2_2.fin, conv2_2.fout, conv2_2.finrow, conv2_2.fincol, conv2_2.frow, conv2_2.fcol, conv2_2.Ksize, conv2_2.Kstride,  
    conv2_2.pad, conv2_2.mask, conv2_2.addr_in, conv2_2.addr_wght, conv2_2.addr_out, conv2_2.pool, conv2_2.relu, 0, 
  conv3_1.fin, conv3_1.fout, conv3_1.finrow, conv3_1.fincol, conv3_1.frow, conv3_1.fcol, conv3_1.Ksize, conv3_1.Kstride,  
    conv3_1.pad, conv3_1.mask, conv3_1.addr_in, conv3_1.addr_wght, conv3_1.addr_out, conv3_1.pool, conv3_1.relu, 0,
  conv3_2.fin, conv3_2.fout, conv3_2.finrow, conv3_2.fincol, conv3_2.frow, conv3_2.fcol, conv3_2.Ksize, conv3_2.Kstride,  
    conv3_2.pad, conv3_2.mask, conv3_2.addr_in, conv3_2.addr_wght, conv3_2.addr_out, conv3_2.pool, conv3_2.relu, 0,
  conv3_3.fin, conv3_3.fout, conv3_3.finrow, conv3_3.fincol, conv3_3.frow, conv3_3.fcol, conv3_3.Ksize, conv3_3.Kstride,  
    conv3_3.pad, conv3_3.mask, conv3_3.addr_in, conv3_3.addr_wght, conv3_3.addr_out, conv3_3.pool, conv3_3.relu, 0,
  conv4_1.fin, conv4_1.fout, conv4_1.finrow, conv4_1.fincol, conv4_1.frow, conv4_1.fcol, conv4_1.Ksize, conv4_1.Kstride,  
    conv4_1.pad, conv4_1.mask, conv4_1.addr_in, conv4_1.addr_wght, conv4_1.addr_out, conv4_1.pool, conv4_1.relu, 0,
  conv4_2.fin, conv4_2.fout, conv4_2.finrow, conv4_2.fincol, conv4_2.frow, conv4_2.fcol, conv4_2.Ksize, conv4_2.Kstride,  
    conv4_2.pad, conv4_2.mask, conv4_2.addr_in, conv4_2.addr_wght, conv4_2.addr_out, conv4_2.pool, conv4_2.relu, 0,
  conv4_3.fin, conv4_3.fout, conv4_3.finrow, conv4_3.fincol, conv4_3.frow, conv4_3.fcol, conv4_3.Ksize, conv4_3.Kstride,  
    conv4_3.pad, conv4_3.mask, conv4_3.addr_in, conv4_3.addr_wght, conv4_3.addr_out, conv4_3.pool, conv4_3.relu, 0,
  conv5_1.fin, conv5_1.fout, conv5_1.finrow, conv5_1.fincol, conv5_1.frow, conv5_1.fcol, conv5_1.Ksize, conv5_1.Kstride,  
    conv5_1.pad, conv5_1.mask, conv5_1.addr_in, conv5_1.addr_wght, conv5_1.addr_out, conv5_1.pool, conv5_1.relu, 0,
  conv5_2.fin, conv5_2.fout, conv5_2.finrow, conv5_2.fincol, conv5_2.frow, conv5_2.fcol, conv5_2.Ksize, conv5_2.Kstride,  
    conv5_2.pad, conv5_2.mask, conv5_2.addr_in, conv5_2.addr_wght, conv5_2.addr_out, conv5_2.pool, conv5_2.relu, 0,
  conv5_3.fin, conv5_3.fout, conv5_3.finrow, conv5_3.fincol, conv5_3.frow, conv5_3.fcol, conv5_3.Ksize, conv5_3.Kstride,  
    conv5_3.pad, conv5_3.mask, conv5_3.addr_in, conv5_3.addr_wght, conv5_3.addr_out, conv5_3.pool, conv5_3.relu, 0
    }; 

  int WT_VOL= (LayerDef+ WEIGHTSIZE );
  float *fdram = (float*)malloc(sizeof(float)* WT_VOL );
  memcpy((void *)(fdram+LayerDef), (const void*)wght_dram, sizeof(float)*WEIGHTSIZE);
  memcpy((void *)(fdram), (const void*)ly_host, sizeof(float)*14*16);

  cl_context       context  = env.getContext();
  cl_kernel        kernel   = env.getKernel();
  cl_command_queue commands = env.getCmdQueue();
  cl_mem           DRAM     = env.getArg0();
    
    cl_event event;
    struct timeval t0, t2;
    int err = clEnqueueWriteBuffer(commands, DRAM, CL_TRUE, 0, sizeof(data_t) * WT_VOL, fdram, 0, NULL, &event);
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
  free(fdram);


  while(1)
  {
    string file ;
    std::getline(out,file);
    if (out.eof()) break;

    cv::Mat img = cv::imread(file, -1);
    CHECK(!img.empty()) << "Unable to decode image " << file;
    std::cout << "---------- Prediction for " << file << " ----------" << std::endl;

    std::vector<Prediction> predictions;
    predictions = classifier.Classify(img, 5);

    /* Print the top N predictions. */
    for (size_t i = 0; i < predictions.size(); ++i) {
      Prediction p = predictions[i];
      std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;
    }
  }
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV



