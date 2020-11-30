#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits>
#include <cmath>

#define LABEL_SIZE		10
#define FEATURE_SIZE	784

void gradient( 
    int num_samples, 
    int L, int D,
    double* weights, 
    double* data, 
    double* output )
{
  //const int L = LABEL_SIZE;
  //const int D = FEATURE_SIZE;
  const int weight_size = (L-1)*D;

  memset(output, 0, sizeof(double)*(weight_size+1));

  double* margins = new double[L-1];

  for(int k = 0; k < num_samples; k++ ) {

    double marginY = 0.0;
    double maxMargin = -std::numeric_limits<double>::infinity();
    int    maxMarginIndex = 0;

    double  label = data[k*(D+1)];
    double* feature = data + k*(D+1) + 1;

    for (int i=0; i<L-1; i++) {
      double margin = 0.0;
      for(int j = 0; j < D; j+=1 ) {
        margin += weights[i*D+j] * feature[j];
      }
      if (i == (int)label - 1) {
        marginY = margin;
      }
      if (margin > maxMargin) {
        maxMargin = margin;
        maxMarginIndex = i;
      }
      margins[i] = margin;
    }

    double sum = 0.0;
    for (int i=0; i<L-1; i++) {
      if (maxMargin > 0) {
        margins[i] -= maxMargin;
        if (i == maxMarginIndex) {
          sum += exp(-maxMargin);
        }
        else {
          sum += exp(margins[i]);
        }
      } 
      else {
        sum += exp(margins[i]);
      }
    }

    // update gradient
    for(int i = 0; i < L-1; i++ ) {
      double multiplier = exp(margins[i]) / (sum+1.0);
      if (label != 0.0 && label == i+1) {
        multiplier -= 1.0;
      }
      for(int j = 0; j < D; j++ ) {
        output[i*D+j] +=  multiplier*feature[j];
      }
    }

    // compute loss
    double loss = log(sum+1); // math.logip(sum)
    if (label > 0.0) {
      loss -= marginY;
    }
    if (maxMargin > 0) {
      loss += maxMargin;
    }
    output[weight_size] += loss;
  }
}

