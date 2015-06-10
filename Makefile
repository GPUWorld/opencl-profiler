all:
	g++ -Wwrite-strings launcher/launcher.cpp -c -ggdb3
	g++ OCL-Performance-Generator.cpp -o ocl-perf-test -lOpenCL launcher.o -lOpenCL -ggdb3
	
clean:
	rm *.o ocl-perf-test