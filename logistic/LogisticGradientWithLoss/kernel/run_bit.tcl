set bit [lindex $argv 0]
create_solution -name $bit -dir . -force
#add_device -vbnv xilinx:adm-pcie-ku3:1ddr:1.1
add_device -vbnv xilinx:adm-pcie-ku3:1ddr:2.0
#add_device -vbnv baidu:leopard-pcie-7v2:fpga_card:0.0

# Kernel Definition
create_kernel gradient -type c
add_files -kernel [get_kernels gradient] "gradient.cpp"

# Define Binary Containers
create_opencl_binary gradient 
set_property region "OCL_REGION_0" [get_opencl_binary gradient]
create_compute_unit -opencl_binary [get_opencl_binary gradient] -kernel [get_kernels gradient] -name k1

# Compile the design for CPU based emulation
#compile_emulation -flow cpu -opencl_binary [get_opencl_binary gradient]

# Compile the application to run on the accelerator card
build_system

# Package the application binaries
#package_system
