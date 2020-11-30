runsim:
	rm -rf exec; mkdir exec;
	cd kernel; make simgen; cd -;
	cp kernel/kernel_top.xclbin exec;
	cd lib_gen; make lib_gen; cd -;
	cp lib_gen/bin/* host;
	cp lib_gen/bin/* exec;
	cd host; make host; cd -;
	cp host/conv_3d_app_acc exec;
	cd exec; XCL_EMULATION_MODE=sw_emu ./conv_3d_app_acc kernel_top.xclbin; cd -;

bitgen:
	rm -rf bit; mkdir bit;
	cd kernel; make bitgen; cd -;

clean:
	cd kernel; make clean; cd -;
	cd lib_gen; make clean; cd -;
	cd host; make clean; cd -;
	cd xilinx_mo; make clean; cd -;
	rm -rf bit exec;
