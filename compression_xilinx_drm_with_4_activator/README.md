# Environment
    Supported Tool: Vitis 2019.2
    Supported Platform: 
        xilinx_u250_xdma_201830_2
        xilinx_u50_gen3x16_xdma_201920_3
    Supported Algorithm: ZLIB level-1

# How to run project:
### Generate xclbin file:
```    
    cd L2/tests/zlib_compress; make step TARGET=hw/hw_emu/sw_emu
```   
### Generate host program binary:
```    
    cd L2/tests/zlib_compress; make host
```    
### Generate estimation report:
```    
    cd L2/tests/zlib_compress; make hls4
``` 
### Run on board:
```    
    ./xil_zlib <input file> <output file>
```    

# Performance:
### 4PE*12engine on xilinx_u250_xdma_201830_2:    
    Pure kernel speed: 7GB/s
    End to end speed: 5GB/s
    
### 3PE*8Engine on xilinx_u50_gen3x16_xdma_201920_3:    
    Pure kernel speed: 7GB/s
    End to end speed: 5GB/s    

### Compression Ratio:
    silesia 2.8
    
# Issues:
### DRM frequency issue:
    After integrated with DRM, if the frequency is over 90MHz, the on board execution will crash.

