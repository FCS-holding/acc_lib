
.PHONY: all

bitgen: step1 step2
simgen: step3 step4

step1:
	v++ -t hw -o conv_3d_kernel.xo -c  __merlinkernel_conv_3d_kernel.cpp -k conv_3d_kernel -s --platform xilinx_u250_xdma_201830_2 -I ../include

step2:
	v++ -t hw -o kernel_top.xclbin --link conv_3d_kernel.xo -s --platform xilinx_u250_xdma_201830_2  -I ../include \
    	--nk conv_3d_kernel:4:conv_3d_kernel0.conv_3d_kernel1.conv_3d_kernel2.conv_3d_kernel3 \
    	--slr conv_3d_kernel0:SLR0 \
		--slr conv_3d_kernel1:SLR1 \
		--slr conv_3d_kernel2:SLR2 \
		--slr conv_3d_kernel3:SLR3 \
		--sp conv_3d_kernel0.m_axi_gmem:DDR[0] \
		--sp conv_3d_kernel1.m_axi_gmem:DDR[1] \
		--sp conv_3d_kernel2.m_axi_gmem:DDR[2] \
		--sp conv_3d_kernel3.m_axi_gmem:DDR[3]

step3:
	v++ -t sw_emu -o conv_3d_kernel.xo -c  __merlinkernel_conv_3d_kernel.cpp -k conv_3d_kernel -s --platform xilinx_u250_xdma_201830_2 -I ../include

step4:
	v++ -t sw_emu -o kernel_top.xclbin --link conv_3d_kernel.xo -s --platform xilinx_u250_xdma_201830_2 -I ../include \
    	--nk conv_3d_kernel:4:conv_3d_kernel0.conv_3d_kernel1.conv_3d_kernel2.conv_3d_kernel3
