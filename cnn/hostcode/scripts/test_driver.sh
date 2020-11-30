#!/bin/bash

if [ "$1" = "valgrind" ] ; then

valgrind --tool=memcheck ./build/examples/cpp_classification_driverTest/classification_fpga.bin models/vgg_model/VGG_ILSVRC_16_layers_deploy.prototxt models/vgg_model/VGG_ILSVRC_16_layers.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt list 

else

./build/examples/cpp_classification_driverTest/classification_fpga.bin models/vgg_model/VGG_ILSVRC_16_layers_deploy.prototxt models/vgg_model/VGG_ILSVRC_16_layers.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt list 

fi
