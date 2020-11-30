
# Environment:<br>
  Supported Tool: Vitis 2019.2<br>
  Supported Platform: xilinx_u250_xdma_201830_2<br>

# How to run:<br>
  Run OpenCL emulation:<br>
  ```
  cd xilinx_prj; make runsim
  ```

  Generate bitstream:<br>
  ```
  cd xilinx_prj; make bitgen
  ```


# Performance:<br>

|IMAGE SIZE      |FILTER SIZE |Time(s)|
|:---------------|:-----------|:------|
|100\*100\*100   |24\*24\*24  |0.053  |
|200\*200\*200   |24\*24\*24  |0.402  |
|400\*400\*400   |24\*24\*24  |2.48   |
|1000\*1000\*1000|24\*24\*24  |36.5   |
