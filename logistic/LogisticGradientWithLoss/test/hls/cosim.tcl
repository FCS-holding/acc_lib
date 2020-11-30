set TOP [file rootname [info script]]

# Project settings
open_project ${TOP} -reset
add_file gradient.cpp -cflags "-DHLS_SIM"
add_file -tb host.cpp 
set_top gradient

# Solution settings
open_solution -reset solution1
set_part xc7vx690tffg1157-2
create_clock -period 240MHz -name default

config_rtl -register_reset

csim_design -argv 32
csynth_design 
cosim_design -argv 32

