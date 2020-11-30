# Environment<br>
    Supported Tool: Vitis 2019.2
    Supported Platform:
        xilinx_u250_xdma_201830_2
        xilinx_u50_gen3x16_xdma_201920_3
    Supported Algorithm: ZLIB level-1


# How to run project:<br>
###  Generate xclbin file:<br>
```
    cd L2/tests/zlib_compress; make step TARGET=hw/hw_emu/sw_emu
```
###  Generate host program binary:<br>
```
    cd L2/tests/zlib_compress; make host<br>
```
###  Generate estimation report:<br>
```
    cd L2/tests/zlib_compress; make hls4
```
###  Run on board:<br>
```
    ./xil_zlib <input file> <output file>
```

# Performance:<br>
###  4PE*12engine on xilinx_u250_xdma_201830_2:<br>
    Pure kernel speed: 7GB/s
    End to end speed: 5GB/s

###  3PE*8Engine on xilinx_u50_gen3x16_xdma_201920_3:<br>
    Pure kernel speed: 7GB/s
    End to end speed: 5GB/s

###    Compression Ratio:
    silesia 2.8<br>
    
# Issues:<br>
### Sub buffer issue:<br>
    This issue could be only reproduced in U50 platform when the sub buffer is larger than 800MB.<br>
    To reproduce the error, we should input a file which size is over 800MB.<br>

