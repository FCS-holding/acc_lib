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

#include "vgg16_sw2.hpp"
#include "/curr/chenzhang/tool/caffe/FPGA/include/kernel-cl.hpp"

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

#define USE_FPGA 1
#define FPGA_Verify 0
#define IMSHOW 0

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);
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
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
#if USE_FPGA
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

std::vector<float> Classifier::FPGA_Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);

  // Forward dimension change to all layers.
  net_->Reshape();
  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);
  Preprocess(img, &input_channels);
/*caffe implementation
  //observe input image data
  //const float* data = net_->input_blobs()[0]->cpu_data();
  //input_channels->at(0).data;
  //unsigned char* chdata = input_channels[0].ptr<uchar>(0);
  //const float* data = net_->input_blobs()[0]->cpu_data(); 
  //uchar* atdata = input_channels.at(0).data;
  //for(int i=0; i<20; i++) {
  //  printf("%f,\t %d\n", data[i], atdata[i]);
  //}
  timeval startSw, endSw;
  gettimeofday(&startSw, NULL);
  //library implementation
  net_->ForwardPrefilled(); 
  //library implementation
  gettimeofday(&endSw, NULL);
  printf("library time :%8.6f ms\n ", (endSw.tv_sec-startSw.tv_sec)*1e+3 + (endSw.tv_usec-startSw.tv_usec)*1e-03 );
*/
/*my software implementation
  float* dram = (float*) malloc(sizeof(float)*(WEIGHTSIZE+INFM+PL53)); //
  float* feat = (float*) malloc(sizeof(float)*(PL53)); //
  float result[1000] = {0};
  memset(dram, 0, sizeof(float)*(WEIGHTSIZE+INFM+PL53)); //
  init_dram(dram);
  printf("init dram finish\n");
  const float* data = net_->input_blobs()[0]->cpu_data();
  uchar* atdata = input_channels.at(0).data;
  for(int i=0; i<INFM; i++) {
    dram[WEIGHTSIZE + i] = (float)data[i];
  }
  printf("write dram with input feature map\n");

  gettimeofday(&startSw, NULL);
  //Software Implementation
  vgg16_sw2(dram); 
  //Software Implementation
  gettimeofday(&endSw, NULL);
  printf("CPU time :%8.6f ms\n ", (endSw.tv_sec-startSw.tv_sec)*1e+3 + (endSw.tv_usec-startSw.tv_usec)*1e-03 );

  //finish feature extraction operation
  memcpy((void*)(feat), (const void*)(dram+(WEIGHTSIZE+INFM)), sizeof(float)*PL53); 
  free(dram);
  //start ann
  ann(feat, result);
  free(feat);
  printf("software finish\n");
*/

  //FPGA Acceleration Test
  printf("FPGA Prediction Start\n");

  int DATA_VOL = (WEIGHTSIZE+ INFM + FM11+FM12+PL12 + FM21+FM22+PL22+ FM31+FM32+FM33+PL33+ FM41+FM42+FM43+PL43+ FM51+FM52+FM53+PL53);
  float *fdram = (float*)malloc(sizeof(float)*DATA_VOL);
  memset(fdram, 0, sizeof(float)*DATA_VOL); //
  float fpga_result[1000];
  
  init_dram(fdram);
  //printf("init fdram finish\n");
  const float* fdata = net_->input_blobs()[0]->cpu_data();
  uchar* fatdata = input_channels.at(0).data;
  for(int i=0; i<INFM; i++) {
    fdram[WEIGHTSIZE + i] = (float)fdata[i];
  }
  //printf("write dram with input feature map\n");
  reorder(fdram);

  //timeval t1, t2, t3;
  //gettimeofday(&t1, NULL);
  //FPGA Kernel Implementation
  kernel_cl(fdram);
  //FPGA Kernel Implementation
  //gettimeofday(&t2, NULL);
  float* fpga_feat = (float*) malloc(sizeof(float)*(PL53)); 
  memcpy(fpga_feat, (const void*)(fdram+DATA_VOL-PL53), sizeof(float)*PL53);
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
  system("CLS");
  system("clear");
  string file = argv[5];

  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;

  cv::Mat img = cv::imread(file, -1);
#if IMSHOW 
  char img_window[] = "Prediction Image";
  cv::imshow(img_window, img);
  cv::moveWindow(img_window, 350, 320);
  cv::waitKey(1000);

  CHECK(!img.empty()) << "Unable to decode image " << file;
  std::vector<Prediction> predictions = classifier.Classify(img);

  /* Print the top N predictions. */
  cv::Mat text_img = cv::Mat::zeros(150,800,CV_8UC3);
  for (size_t i = 0; i < predictions.size(); ++i) {
    Prediction p = predictions[i];
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;

    std::stringstream msg;
    msg << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"";
    std::string text = msg.str();//"Funny text inside the box";
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.5;
    int thickness = 1;  
    cv::Point textOrg(20, 20+i*20);
    cv::putText(text_img, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness,8);
  }
  char txt_window[] = "Classification Result";
  imshow(txt_window, text_img);
  cv::moveWindow(txt_window, 200, 720);

  cv::waitKey(0);
#else
  CHECK(!img.empty()) << "Unable to decode image " << file;
  std::vector<Prediction> predictions = classifier.Classify(img);

  /* Print the top N predictions. */
  for (size_t i = 0; i < predictions.size(); ++i) {
    Prediction p = predictions[i];
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;
  }
#endif

}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV



