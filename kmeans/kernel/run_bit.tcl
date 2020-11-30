set bit [lindex $argv 0]
create_solution -name myproj -dir . -force
#add_device -vbnv xilinx:adm-pcie-7v3:1ddr:1.1
#add_device -vbnv xilinx:adm-pcie-ku3:1ddr:1.1
add_device -vbnv xilinx:adm-pcie-ku3:1ddr:2.0

# Kernel Definition
create_kernel kmeans -type c
add_files -kernel [get_kernels kmeans] "kmeans.cpp"

# Define Binary Containers
create_opencl_binary kmeans 
set_property region "OCL_REGION_0" [get_opencl_binary kmeans]
create_compute_unit -opencl_binary [get_opencl_binary kmeans] -kernel [get_kernels kmeans] -name k1

# Compile the design for CPU based emulation
#compile_emulation -flow cpu -opencl_binary [get_opencl_binary kmeans]

# Compile the application to run on the accelerator card
build_system

# Package the application binaries
#package_system
