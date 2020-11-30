Note: kmeans kernel currently ignores the norm data, so kmeans_base function in 
baseline.h has been changed to ignore the norm data by default to match the 
kmeans kernel. Pass INCLUDE_NORM macro if norm data processing is needed in
kmeans_base.

# OpenCL Standalone Test

$cd test/ocl

## Softare Emulation
```
$make clean; make run_sw_emu
XCL_EMULATION_MODE=true ./host_top kmeans_sw_emu.xclbin 1
FPGA execution takes 5.4050 ms
FPGA execution takes 3.7540 ms
FPGA execution takes 3.7610 ms
diff: 0.000000 max, 0.000000/point, 0.000000%/point
```

## Hardware Emulation
```
$make clean; make run_hw_emu`
```

## Hardware Run
```
$make clean;make bit`
$./host_top kmeans_hw.xclbin 1
FPGA execution takes 1.9520 ms
FPGA execution takes 1.7600 ms
FPGA execution takes 1.7560 ms
diff: 0.000000 max, 0.000000/point, 0.000000%/point
```

# HLS Standalone Test
```
$cd test/hls
$vivado_hls -f run.tcl
``` 



