set top fastSquareDistance
set top kmeans

# Project settings
open_project proj -reset
add_files kmeans.cpp -cflags "-DHLS_SIM"
add_files -tb host.cpp -cflags "-DHLS_SIM"
set_top $top

# Solution settings
open_solution -reset solution1
#set_part xc7vx690tffg1157-2
set_part xcku060-ffva1156-2-e
create_clock -period 240MHz

csim_design -argv 1
exit
#config_rtl -register_reset
#config_array_partition -throughput_driven
#csynth_design
#cosim_design -argv "49 9 3" -bc
#cosim_design -argv "49 9 3"

