create_solution -name hw_emu -dir . -force
add_device -vbnv xilinx:adm-pcie-7v3:1ddr:1.0

# Host Compiler Flags
set_property -name host_cflags -value "-m64 -std=c++0x -DBLAZE_TEST" -objects [current_project]

# Host Source Files
add_files "host.cpp"
add_files "gradient.cpp"

# Kernel Definition
create_kernel gradient -type c
add_files -kernel [get_kernels gradient] "gradient.cpp"

# Define Binary Containers
create_opencl_binary gradient 
set_property region "OCL_REGION_0" [get_opencl_binary gradient]
create_compute_unit -opencl_binary [get_opencl_binary gradient] -kernel [get_kernels gradient] -name k1

# Compile the design for CPU based emulation
compile_emulation -flow hardware -opencl_binary [get_opencl_binary gradient]

# Run the compiled application in CPU based emulation mode
run_emulation -flow hardware -args "gradient.xclbin 32"
