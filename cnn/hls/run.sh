rm a.out *.txt
#/curr/software/Xilinx/SDAccel/2015.3/lnx64/tools/gcc/bin/g++ -O2 -I/curr/software/Xilinx/Vivado_HLS/2015.3/include/  main.cpp vgg16.cpp vgg16_sw2.cpp
g++ -O2 -I/curr/software/Xilinx/Vivado_HLS/2015.3/include/  main.cpp vgg16.cpp vgg16_sw2.cpp

if [ "$1" = valgrind ]; then
valgrind --tool=memcheck --leak-check=full ./a.out
else
./a.out
fi
