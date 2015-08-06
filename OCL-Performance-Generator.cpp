/*
 ============================================================================
 Name        : OCL-Performance-Generator.cpp
 Author      : Diego Nieto
 Version     :
 Copyright   : GNU License
 Description : Hello World in C++,
 ============================================================================
 */

#include <iostream>
#include <CL/cl.h>

#include "launcher/launcher.hpp"
#include "launcher/execution.hpp"

using namespace std;

int main(void) {
	Launcher launcher(2,64,1,2);

	launcher.initKernel("vadd.cl","vadd");

	int err;

	int size = 4;
	unsigned int totalBytes = (sizeof(float)*4);

  size_t globalWorkSize[1];
  globalWorkSize[0] = 1024;

  launcher.setArgs(OclArg(totalBytes, BUFFER, CL_MEM_READ_ONLY),
  								 OclArg(totalBytes, BUFFER, CL_MEM_READ_ONLY),
  								 OclArg(totalBytes, BUFFER, CL_MEM_READ_WRITE),
  								 OclArg(sizeof(cl_int), OTHER, NULL, (void**)&size));

  launcher.launch(globalWorkSize);

  Execution *best = launcher.getBest();

	cout << "Number of executions: " << launcher.getNumberOfExecutions() << endl; /* prints Hello World */
	cout << "Best time: " << best->getTime() << ". x=" << best->getX() << ", y=" << best->getY() << endl;
	return 0;
}
