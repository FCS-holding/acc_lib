#*******************************************************************************
# Define the solution for SDAccel
create_solution -name example_alpha -dir . -force
add_device -vbnv xilinx:adm-pcie-ku3:1ddr:2.0

# Host Compiler Flags
set_property -name host_cflags -value "-g -Wall -D FPGA_DEVICE -D C_KERNEL" -objects [current_solution]

# Host Source Files
add_files "test-cl.c"

# Kernel Definition
create_kernel mmult -type c
add_files -kernel [get_kernels mmult] "mmult1.c"

# Define Binary Containers
create_opencl_binary mmult1
set_property region "OCL_REGION_0" [get_opencl_binary mmult1]
create_compute_unit -opencl_binary [get_opencl_binary mmult1] -kernel [get_kernels mmult] -name k1

# Compile the design for CPU based emulation
compile_emulation -flow cpu -opencl_binary [get_opencl_binary mmult1]

# Run the compiled application in CPU based emulation mode
run_emulation -flow cpu -args "mmult1.xclbin"

compile_emulation -flow hardware -opencl_binary [get_opencl_binary mmult1]
run_emulation -flow hardware -args "mmult1.xclbin"

# Compile the application to run on the accelerator card
# build_system

# Package the application binaries
# package_system

