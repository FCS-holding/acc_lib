rm a.out *.txt
#/curr/software/Xilinx/SDAccel/2015.3/lnx64/tools/gcc/bin/g++ -O2 -I/curr/software/Xilinx/Vivado_HLS/2015.3/include/  main.cpp vgg16.cpp vgg16_sw2.cpp

if [ "$1" = half ]; then
g++ -O2 -I/curr/software/Xilinx/Vivado_HLS/2015.3/include/ -DHLS_NO_XIL_FPO_LIB main.cpp vgg16.cpp vgg16_sw2.cpp
else
g++ -O2 -I/curr/software/Xilinx/Vivado_HLS/2015.3/include/ main.cpp vgg16.cpp vgg16_sw2.cpp
fi
./a.out 
