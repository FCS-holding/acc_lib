
PIC=examples/images/cat.jpg
#FLAG=exec
if [ ! -z $2 ]; then
  PIC=$2;
fi

# rm result.txt
# export OPENBLAS_NUM_THREADS=12

if [ "$1" = "valgrind" ]; then
	valgrind --tool=memcheck --leak-check=full ./build/examples/cpp_classification/classification.bin models/vgg_model/VGG_ILSVRC_16_layers_deploy.prototxt models/vgg_model/VGG_ILSVRC_16_layers.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt examples/images/cat.jpg
elif [ "$1" = "cpu" ]; then
	./build/examples/cpp_classification/classification.bin models/vgg_model/VGG_ILSVRC_16_layers_deploy.prototxt models/vgg_model/VGG_ILSVRC_16_layers.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt $PIC
elif [ "$1" = "fpga" ]; then
	./build/examples/cpp_classification/classification_fpga.bin models/vgg_model/VGG_ILSVRC_16_layers_deploy.prototxt models/vgg_model/VGG_ILSVRC_16_layers.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt $PIC 
else 
   echo "Commandline: sh test_run.sh [device](cpu/fpga) [path to image](default: examples/images/cat.jpg) " 
fi

