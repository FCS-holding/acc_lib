#!/bin/bash

echo "mode: "$2
echo "device: " $1

if [ "$2" = "batch" ]; then
		if [ "$1" == "cpu" ]; then
			./build/examples/cpp_classification/classification_batch.bin models/vgg_model/VGG_ILSVRC_16_layers_deploy.prototxt models/vgg_model/VGG_ILSVRC_16_layers.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt list
	        #sleep 3
		elif [ "$1" == "fpga" ]; then
			./build/examples/cpp_classification/classification_fpga_batch.bin models/vgg_model/VGG_ILSVRC_16_layers_deploy.prototxt models/vgg_model/VGG_ILSVRC_16_layers.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt list 
	        #sleep 3
        elif [ "$1" == "fpga_sda" ]; then
			./build/examples/cpp_classification/classification_fpga_batch_sda.bin models/vgg_model/VGG_ILSVRC_16_layers_deploy.prototxt models/vgg_model/VGG_ILSVRC_16_layers.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt list 
	        #sleep 3
        elif [ "$1" == "fpga_half" ]; then
			./build/examples/cpp_classification_fpga_half/classification_fpga_half.bin models/vgg_model/VGG_ILSVRC_16_layers_deploy.prototxt models/vgg_model/VGG_ILSVRC_16_layers.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt list 
        elif [ "$1" == "fpga_fx" ]; then
			./build/examples/cpp_classification_fpga_fx/classification_fpga.bin models/vgg_model/VGG_ILSVRC_16_layers_deploy.prototxt models/vgg_model/VGG_ILSVRC_16_layers.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt list 
        elif [ "$1" == "fpga_fxdyn" ]; then
			./build/examples/cpp_classification_fpga_fxdyn/classification_fpga.bin models/vgg_model/VGG_ILSVRC_16_layers_deploy.prototxt models/vgg_model/VGG_ILSVRC_16_layers.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt list 
	        #sleep 3
		else 

		   echo "Commandline: sh test_run.sh [device](cpu/fpga) [path to image](default: examples/images/cat.jpg) " 
		fi
else
	file=list
	cat $file | while read line
	do
	    clear
	    echo $line
		if [ "$1" == "cpu" ]; then
			./build/examples/cpp_classification/classification.bin models/vgg_model/VGG_ILSVRC_16_layers_deploy.prototxt models/vgg_model/VGG_ILSVRC_16_layers.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt $line
	        #sleep 3
		elif [ "$1" == "fpga" ]; then
			./build/examples/cpp_classification/classification_fpga.bin models/vgg_model/VGG_ILSVRC_16_layers_deploy.prototxt models/vgg_model/VGG_ILSVRC_16_layers.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt $line 
	        #sleep 3
		else 
		   echo "Commandline: sh test_run.sh [device](cpu/fpga) [path to image](default: examples/images/cat.jpg) " 
		fi
	done
fi
